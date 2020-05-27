use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

use async_trait::async_trait;

use crate::error;
use crate::internal::block::Store;
use crate::state::{Collection, Derived, State};
use crate::transaction::Txn;
use crate::value::link::TCPath;
use crate::value::{TCResult, TCValue, ValueId};

pub struct Slice;

impl TryFrom<TCValue> for Slice {
    type Error = error::TCError;

    fn try_from(_value: TCValue) -> TCResult<Slice> {
        Err(error::not_implemented())
    }
}

struct BTree {
    blocks: Arc<Store>,
}

pub struct IndexConfig {
    key: HashMap<ValueId, TCPath>,
    values: HashMap<ValueId, TCPath>,
}

impl TryFrom<TCValue> for IndexConfig {
    type Error = error::TCError;

    fn try_from(value: TCValue) -> TCResult<IndexConfig> {
        let (key, values): (TCValue, TCValue) = value.try_into()?;
        let key: Vec<(ValueId, TCPath)> = key.try_into()?;
        let values: Vec<(ValueId, TCPath)> = values.try_into()?;
        Ok(IndexConfig {
            key: key.into_iter().collect(),
            values: values.into_iter().collect(),
        })
    }
}

pub struct Index {
    config: IndexConfig,
    data: BTree,
}

impl TryFrom<TCValue> for Index {
    type Error = error::TCError;

    fn try_from(_value: TCValue) -> TCResult<Index> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl Collection for Index {
    type Key = Slice;
    type Value = Index;

    async fn get(self: &Arc<Self>, _txn: &Arc<Txn<'_>>, _key: &Slice) -> TCResult<Index> {
        Err(error::not_implemented())
    }

    async fn put(
        self: Arc<Self>,
        _txn: &Arc<Txn<'_>>,
        _key: Slice,
        _value: Index,
    ) -> TCResult<State> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl Derived for Index {
    type Config = IndexConfig;

    async fn create(txn: &Arc<Txn<'_>>, config: Self::Config) -> TCResult<Arc<Self>> {
        let data = BTree {
            blocks: txn.context(),
        };
        Ok(Arc::new(Index { config, data }))
    }
}
