use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

use async_trait::async_trait;

use crate::error;
use crate::internal::FsDir;
use crate::state::{Collection, Derived};
use crate::transaction::Transaction;
use crate::value::{Link, TCResult, TCValue};

struct Index {
    fs_dir: FsDir,
}

#[async_trait]
impl Collection for Index {
    type Key = Vec<TCValue>;
    type Value = Vec<TCValue>;

    async fn get(
        self: &Arc<Self>,
        _txn: Arc<Transaction>,
        _key: &Self::Key,
    ) -> TCResult<Self::Value> {
        Err(error::not_implemented())
    }

    async fn put(
        self: Arc<Self>,
        _txn: Arc<Transaction>,
        _key: Self::Key,
        _value: Self::Value,
    ) -> TCResult<Arc<Self>> {
        Err(error::not_implemented())
    }
}

struct IndexConfig {
    key: Vec<(String, Link)>,
    value: Vec<(String, Link)>,
}

impl TryFrom<TCValue> for IndexConfig {
    type Error = error::TCError;

    fn try_from(value: TCValue) -> TCResult<IndexConfig> {
        let (key, value): (TCValue, TCValue) = value.try_into()?;
        Ok(IndexConfig {
            key: key.try_into()?,
            value: value.try_into()?,
        })
    }
}

#[async_trait]
impl Derived for Index {
    type Config = IndexConfig;

    fn from(_source: impl Collection, _config: IndexConfig) -> TCResult<Arc<Index>> {
        Err(error::not_implemented())
    }
}
