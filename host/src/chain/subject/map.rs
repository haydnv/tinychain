use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::iter::FromIterator;
use std::sync::Arc;

use async_trait::async_trait;
use destream::de;
use futures::future::{self, join_all, TryFutureExt};
use futures::stream::{FuturesUnordered, StreamExt, TryStreamExt};
use futures::{join, try_join};
use log::{debug, warn};
use safecast::{CastFrom, TryCastFrom, TryCastInto};
use sha2::digest::Output;
use sha2::Sha256;
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use tc_error::*;
use tc_transact::fs::{Dir, DirRead, DirWrite, File, FileRead, FileWrite};
use tc_transact::lock::{TxnLock, TxnLockReadGuard, TxnLockWriteGuard};
use tc_transact::{IntoView, Transact, Transaction, TxnId};
use tcgeneric::{Id, Map, TCBoxTryFuture, Tuple};

use crate::collection::Collection;
use crate::fs;
use crate::fs::FileReadGuard;
use crate::scalar::{Scalar, ScalarType, TCRef};
use crate::state::{State, StateView};
use crate::txn::Txn;

use super::{CollectionSchema, SubjectCollection, DYNAMIC};

const BLOCK_SIZE_HINT: usize = 4096;
const LOCK_NAME: &str = "SubjectMap keys";

/// The `Subject` of a `Chain` which is a `Map` of `Collection`s
#[derive(Clone)]
pub struct SubjectMap {
    dir: fs::Dir,
    // TODO: is there a way to avoid storing each `Id` twice?
    ids: TxnLock<BTreeSet<Id>>,
    collections: Arc<RwLock<Map<SubjectCollection>>>,
}

impl SubjectMap {
    async fn read(
        self,
        txn_id: TxnId,
    ) -> TCResult<(
        TxnLockReadGuard<BTreeSet<Id>>,
        OwnedRwLockReadGuard<Map<SubjectCollection>>,
    )> {
        let ids = self.ids.read(txn_id).await?;
        let collections = self.collections.read_owned().await;
        Ok((ids, collections))
    }

    async fn write(
        self,
        txn_id: TxnId,
    ) -> TCResult<(
        TxnLockWriteGuard<BTreeSet<Id>>,
        OwnedRwLockWriteGuard<Map<SubjectCollection>>,
    )> {
        let ids = self.ids.write(txn_id).await?;
        let collections = self.collections.write_owned().await;
        Ok((ids, collections))
    }

    pub async fn create(dir: fs::Dir, txn_id: TxnId) -> TCResult<Self> {
        let mut dir_lock = dir.write(txn_id).await?;

        if !dir_lock.is_empty() {
            return Err(TCError::internal(
                "tried to create a new dynamic Chain with a non-empty directory",
            ));
        }

        let _file: fs::File<Scalar> =
            dir_lock.create_file(DYNAMIC.into(), ScalarType::default())?;

        Ok(Self {
            dir,
            ids: TxnLock::new(LOCK_NAME, BTreeSet::new()),
            collections: Arc::new(RwLock::new(Map::new())),
        })
    }

    pub(super) fn load(txn: &Txn, dir: fs::Dir) -> TCBoxTryFuture<Self> {
        Box::pin(async move {
            let txn_id = *txn.id();

            let file: fs::FileReadGuard<Scalar> = {
                let dir = dir.read(txn_id).await?;
                let file = dir.get_file(&DYNAMIC.into())?;
                let file: fs::File<Scalar> = file.ok_or_else(|| {
                    TCError::internal(format!("dynamic Chain is missing its schema file"))
                })?;

                file.read(txn_id).await?
            };

            let mut map = Map::new();
            for id in FileReadGuard::<Scalar>::block_ids(&file) {
                let schema = file.read_block(id).await?;
                let schema = Scalar::clone(&*schema);
                let schema = schema.try_cast_into(|s| {
                    TCError::internal(format!("invalid schema for dynamic Chain subject: {}", s))
                })?;

                let schema = CollectionSchema::from_scalar(schema)?;
                let subject = SubjectCollection::load(txn, schema, &dir, id.clone()).await?;

                map.insert(id.clone(), subject);
            }

            Ok(Self {
                dir,
                ids: TxnLock::new(LOCK_NAME, map.keys().cloned().collect()),
                collections: Arc::new(RwLock::new(map)),
            })
        })
    }

    pub(super) fn restore<'a>(
        &'a self,
        txn: &'a Txn,
        backups: Map<Collection>,
    ) -> TCBoxTryFuture<()> {
        Box::pin(async move {
            let container = self.dir.clone();

            let (mut ids, mut collections) = self.clone().write(*txn.id()).await?;

            for id in collections.keys() {
                if !backups.contains_key(id) {
                    warn!("backup of dynamic Chain is missing collection {}", id);
                }
            }

            let restores = FuturesUnordered::new();
            for (id, backup) in backups.into_iter() {
                if let Some(collection) = collections.get(&id).cloned() {
                    assert!(ids.contains(&id));
                    restores.push(async move { collection.restore(txn, backup).await });
                } else {
                    // TODO: parallelize
                    ids.insert(id.clone());
                    put(txn, &container, &mut collections, id, backup).await?;
                }
            }

            restores.try_fold((), |(), ()| future::ready(Ok(()))).await
        })
    }

    pub async fn get(self, txn_id: TxnId, id: &Id) -> TCResult<Option<SubjectCollection>> {
        let (ids, collections) = self.read(txn_id).await?;
        debug!(
            "get {} from SubjectMap with members {}",
            id,
            Tuple::<&Id>::from_iter(ids.iter())
        );

        if ids.contains(id) {
            Ok((**collections).get(id).cloned())
        } else {
            Ok(None)
        }
    }

    pub async fn put(self, txn: &Txn, id: Id, collection: Collection) -> TCResult<()> {
        let txn_id = *txn.id();
        let container = self.dir.clone();

        let (mut ids, mut collections) = self.write(txn_id).await?;

        if let Some(existing) = collections.get(&id) {
            assert!(ids.contains(&id));
            let (existing_hash, collection_hash) = try_join!(
                collection.clone().hash(txn.clone()),
                existing.clone().hash(txn.clone())
            )?;

            if existing_hash == collection_hash {
                Ok(())
            } else {
                put(txn, &container, &mut collections, id, collection).await
            }
        } else {
            ids.insert(id.clone());
            put(txn, &container, &mut collections, id, collection).await
        }
    }

    pub async fn into_state(self, txn_id: TxnId) -> TCResult<State> {
        let (_ids, collections) = self.read(txn_id).await?;
        let mut state = Map::new();

        for (id, collection) in collections.iter() {
            let collection = collection.clone().into_state(txn_id).await?;
            state.insert(id.clone(), collection);
        }

        Ok(State::Map(state))
    }

    pub async fn hash(self, txn: Txn) -> TCResult<Output<Sha256>> {
        let (_ids, collections) = self.read(*txn.id()).await?;

        let collections = collections
            .iter()
            .map(|(id, collection)| {
                let collection = State::Collection(Collection::from((&*collection).clone()));
                (id.clone(), collection)
            })
            .collect();

        State::Map(collections).hash(txn).await
    }
}

#[async_trait]
impl Transact for SubjectMap {
    type Commit = ();

    async fn commit(&self, txn_id: &TxnId) {
        join!(self.dir.commit(txn_id), self.ids.commit(txn_id));

        let collections = self.collections.read().await;

        let mut commits = Vec::with_capacity(collections.len());
        for collection in collections.values() {
            commits.push(collection.commit(txn_id));
        }

        join_all(commits).await;
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join!(self.dir.finalize(txn_id), self.ids.finalize(txn_id));

        let collections = self.collections.read().await;
        let finalized: FuturesUnordered<_> =
            collections.values().map(|c| c.finalize(txn_id)).collect();

        finalized.fold((), |(), ()| future::ready(())).await
    }
}

#[async_trait]
impl de::FromStream for SubjectMap {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(
        txn: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        decoder.decode_map(SubjectMapVisitor { txn }).await
    }
}

struct SubjectMapVisitor {
    txn: Txn,
}

#[async_trait]
impl de::Visitor for SubjectMapVisitor {
    type Value = SubjectMap;

    fn expecting() -> &'static str {
        "a Map of Collections"
    }

    async fn visit_map<A: de::MapAccess>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let txn_id = *self.txn.id();
        let mut collections = Map::new();

        let mut file: fs::FileWriteGuard<Scalar> = {
            let file: fs::File<Scalar> = {
                let mut dir = self
                    .txn
                    .context()
                    .write(txn_id)
                    .map_err(de::Error::custom)
                    .await?;

                dir.create_file(DYNAMIC.into(), ScalarType::default())
                    .map_err(de::Error::custom)?
            };

            file.write(txn_id).map_err(de::Error::custom).await?
        };

        while let Some(key) = map.next_key::<Id>(()).await? {
            let txn = self
                .txn
                .subcontext(key.clone())
                .map_err(de::Error::custom)
                .await?;

            let collection: SubjectCollection = map.next_value(txn).await?;

            let schema = Scalar::cast_from(collection.schema());
            file.create_block(key.clone(), schema, BLOCK_SIZE_HINT)
                .map_err(de::Error::custom)
                .await?;

            collections.insert(key, collection);
        }

        Ok(SubjectMap {
            dir: self.txn.context().clone(),
            ids: TxnLock::new(LOCK_NAME, collections.keys().cloned().collect()),
            collections: Arc::new(RwLock::new(collections)),
        })
    }
}

#[async_trait]
impl<'en> IntoView<'en, fs::Dir> for SubjectMap {
    type Txn = Txn;
    type View = StateView<'en>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        let (_ids, collections) = self.read(*txn.id()).await?;

        let mut map = HashMap::with_capacity(collections.len());
        for (id, collection) in collections.iter() {
            let view = Collection::from((&*collection).clone())
                .into_view(txn.clone())
                .await?;

            map.insert(id.clone(), StateView::Collection(view));
        }

        Ok(StateView::Map(map))
    }
}

impl fmt::Debug for SubjectMap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for SubjectMap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a Map of Collections")
    }
}

async fn put(
    txn: &Txn,
    container: &fs::Dir,
    collections: &mut Map<SubjectCollection>,
    id: Id,
    collection: Collection,
) -> TCResult<()> {
    let txn_id = *txn.id();
    let mut schema = {
        let container = container.read(txn_id).await?;
        let schema: Option<fs::File<Scalar>> = container.get_file(&DYNAMIC.into())?;
        let schema = schema.ok_or_else(|| TCError::internal("missing schema file"))?;
        schema.write(txn_id).await?
    };

    let collection = if let Some(existing) = collections.get(&id) {
        existing.restore(txn, collection).await?;
        existing.clone()
    } else {
        let collection = Collection::copy_from(txn, container, id.clone(), collection).await?;
        SubjectCollection::from_collection(collection)?
    };

    if schema.contains(&id) {
        let block = schema.read_block(&id).await?;
        let tc_ref = TCRef::try_cast_from((*block).clone(), |block| {
            TCError::internal(format!("bad schema for subject map collection: {}", block))
        })?;

        let actual = CollectionSchema::from_scalar(tc_ref)?;
        if collection.schema() != actual {
            return Err(TCError::unsupported(format!(
                "cannot change schema of {} from {} to {}",
                collection, collection, actual
            )));
        }
    } else {
        schema
            .create_block(
                id.clone(),
                Scalar::cast_from(collection.schema()),
                BLOCK_SIZE_HINT,
            )
            .await?;
    }

    collections.insert(id, collection);
    Ok(())
}
