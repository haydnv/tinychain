use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::iter::FromIterator;
use std::sync::Arc;

use async_trait::async_trait;
use destream::de;
use futures::future::{self, TryFutureExt};
use futures::join;
use futures::stream::{FuturesUnordered, StreamExt};
use futures::TryStreamExt;
use log::{debug, warn};
use safecast::{CastFrom, CastInto, TryCastInto};
use sha2::digest::Output;
use sha2::Sha256;
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use tc_error::*;
use tc_transact::fs::{Dir, File, Store};
use tc_transact::lock::{TxnLock, TxnLockReadGuard, TxnLockWriteGuard};
use tc_transact::{IntoView, Transact, Transaction, TxnId};
use tcgeneric::{Id, Map, TCBoxTryFuture, Tuple};

use crate::collection::Collection;
use crate::fs;
use crate::scalar::{Scalar, ScalarType};
use crate::state::{State, StateView};
use crate::txn::Txn;

use super::{CollectionSchema, SubjectCollection, DYNAMIC};

const BLOCK_SIZE_HINT: usize = 4096;
const LOCK_NAME: &str = "SubjectMap keys";

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
        if !dir.is_empty(txn_id).await? {
            return Err(TCError::internal(
                "tried to create a new dynamic Chain with a non-empty directory",
            ));
        }

        let _file: fs::File<Scalar> = dir
            .create_file(txn_id, DYNAMIC.into(), ScalarType::default())
            .await?;

        Ok(Self {
            dir,
            ids: TxnLock::new(LOCK_NAME, BTreeSet::new()),
            collections: Arc::new(RwLock::new(Map::new())),
        })
    }

    pub(super) fn load(txn: &Txn, dir: fs::Dir) -> TCBoxTryFuture<Self> {
        Box::pin(async move {
            let txn_id = *txn.id();

            let file = dir.get_file(txn_id, &DYNAMIC.into()).await?;
            let file: fs::File<Scalar> = file.ok_or_else(|| {
                TCError::internal(format!("dynamic Chain is missing its schema file"))
            })?;

            let mut map = Map::new();
            for id in file.block_ids(txn_id).await? {
                let schema = file.read_block(txn_id, id.clone()).await?;
                let schema = Scalar::clone(&*schema);
                let schema = schema.try_cast_into(|s| {
                    TCError::internal(format!("invalid schema for dynamic Chain subject: {}", s))
                })?;

                let schema = CollectionSchema::from_scalar(schema)?;
                let subject = SubjectCollection::load(txn, schema, &dir, id.clone()).await?;

                map.insert(id, subject);
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
            let txn_id = *txn.id();
            let container = self.dir.clone();
            let schema = container.get_file(txn_id, &DYNAMIC.into()).await?;
            let schema: fs::File<Scalar> =
                schema.ok_or_else(|| TCError::internal("missing schema file"))?;

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
                    put(txn, &schema, &container, &mut collections, id, backup).await?;
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
            Ok(collections.get(id).cloned())
        } else {
            Ok(None)
        }
    }

    pub async fn put(self, txn: &Txn, id: Id, collection: Collection) -> TCResult<()> {
        let container = self.dir.clone();
        let txn_id = *txn.id();
        let file = self
            .dir
            .get_file::<fs::File<Scalar>, Scalar>(txn_id, &DYNAMIC.into())
            .await?;

        let file = file.ok_or_else(|| TCError::internal("dynamic Chain missing schema file"))?;

        let (mut ids, mut collections) = self.write(txn_id).await?;
        if ids.contains(&id) {
            Err(TCError::bad_request(
                "SubjectMap already contains an entry called",
                id,
            ))
        } else {
            ids.insert(id.clone());
            put(txn, &file, &container, &mut collections, id, collection).await
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
    async fn commit(&self, txn_id: &TxnId) {
        join!(self.dir.commit(txn_id), self.ids.commit(txn_id));

        let collections = self.collections.read().await;
        let commits: FuturesUnordered<_> = collections.values().map(|c| c.commit(txn_id)).collect();
        commits.fold((), |(), ()| future::ready(())).await
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

        let dir = self.txn.context().clone();
        let file: fs::File<Scalar> = dir
            .create_file(txn_id, DYNAMIC.into(), ScalarType::default())
            .map_err(de::Error::custom)
            .await?;

        while let Some(key) = map.next_key::<Id>(()).await? {
            let txn = self
                .txn
                .subcontext(key.clone())
                .map_err(de::Error::custom)
                .await?;

            let collection: SubjectCollection = map.next_value(txn).await?;

            let schema = Scalar::cast_from(collection.schema());
            file.create_block(txn_id, key.clone(), schema, BLOCK_SIZE_HINT)
                .map_err(de::Error::custom)
                .await?;

            collections.insert(key, collection);
        }

        Ok(SubjectMap {
            dir,
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
    schema: &fs::File<Scalar>,
    container: &fs::Dir,
    collections: &mut Map<SubjectCollection>,
    id: Id,
    collection: Collection,
) -> TCResult<()> {
    let txn_id = *txn.id();
    let collection = Collection::copy_from(txn, container, id.clone(), collection).await?;

    let collection = SubjectCollection::from_collection(collection)?;
    schema
        .create_block(
            txn_id,
            id.clone(),
            collection.schema().cast_into(),
            BLOCK_SIZE_HINT,
        )
        .await?;

    collections.insert(id, collection);

    Ok(())
}
