use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use destream::de;
use sha2::digest::Output;
use sha2::Sha256;
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use tc_error::*;
use tc_transact::lock::{TxnLock, TxnLockReadGuard, TxnLockWriteGuard};
use tc_transact::{IntoView, Transaction, TxnId};
use tcgeneric::{Id, Map};

use crate::fs;
use crate::state::State;
use crate::txn::Txn;

use super::{Collection, CollectionView};

#[derive(Clone)]
pub struct CollectionMap {
    // TODO: is there a way to avoid storing each `Id` twice?
    ids: TxnLock<BTreeSet<Id>>,
    collections: Arc<RwLock<Map<Collection>>>,
}

impl CollectionMap {
    async fn read(
        self,
        txn_id: TxnId,
    ) -> TCResult<(
        TxnLockReadGuard<BTreeSet<Id>>,
        OwnedRwLockReadGuard<Map<Collection>>,
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
        OwnedRwLockWriteGuard<Map<Collection>>,
    )> {
        let ids = self.ids.write(txn_id).await?;
        let collections = self.collections.write_owned().await;
        Ok((ids, collections))
    }

    pub async fn get(self, txn_id: TxnId, id: &Id) -> TCResult<Option<Collection>> {
        let (ids, collections) = self.read(txn_id).await?;
        if ids.contains(id) {
            Ok(collections.get(id).cloned())
        } else {
            Ok(None)
        }
    }

    pub async fn put(self, txn_id: TxnId, id: Id, collection: Collection) -> TCResult<()> {
        let (ids, mut collections) = self.write(txn_id).await?;
        if ids.contains(&id) {
            Err(TCError::bad_request(
                "CollectionMap already contains an entry called",
                id,
            ))
        } else {
            collections.insert(id, collection);
            Ok(())
        }
    }

    pub async fn hash(self, txn: Txn) -> TCResult<Output<Sha256>> {
        let (_ids, collections) = self.read(*txn.id()).await?;

        let collections = collections
            .iter()
            .map(|(id, collection)| {
                let collection = State::Collection(Collection::clone(&*collection));
                (id.clone(), collection)
            })
            .collect();

        State::Map(collections).hash(txn).await
    }
}

impl From<Map<Collection>> for CollectionMap {
    fn from(collections: Map<Collection>) -> Self {
        let ids = TxnLock::new(
            "CollectionMap key set",
            collections.keys().cloned().collect(),
        );

        let collections = Arc::new(RwLock::new(collections));
        Self { ids, collections }
    }
}

#[async_trait]
impl de::FromStream for CollectionMap {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(
        txn: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        decoder.decode_map(CollectionMapVisitor { txn }).await
    }
}

struct CollectionMapVisitor {
    txn: Txn,
}

#[async_trait]
impl de::Visitor for CollectionMapVisitor {
    type Value = CollectionMap;

    fn expecting() -> &'static str {
        "a Map of Collections"
    }

    async fn visit_map<A: de::MapAccess>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut collections = Map::new();

        while let Some(key) = map.next_key(()).await? {
            let collection = map.next_value(self.txn.clone()).await?;
            collections.insert(key, collection);
        }

        Ok(CollectionMap::from(collections))
    }
}

#[async_trait]
impl<'en> IntoView<'en, fs::Dir> for CollectionMap {
    type Txn = Txn;
    type View = Map<CollectionView<'en>>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        let (_ids, collections) = self.read(*txn.id()).await?;

        let mut map = Map::new();
        for (id, collection) in collections.iter() {
            let view = Collection::clone(&*collection)
                .into_view(txn.clone())
                .await?;

            map.insert(id.clone(), view);
        }

        Ok(map)
    }
}

impl fmt::Debug for CollectionMap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for CollectionMap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a Map of Collections")
    }
}
