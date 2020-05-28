use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

use async_trait::async_trait;

use crate::error;
use crate::i18n::Locale;
use crate::internal::block::Store;
use crate::state::{Collection, Derived, State};
use crate::transaction::{Txn, TxnId};
use crate::value::link::TCPath;
use crate::value::op::PutOp;
use crate::value::{TCResult, TCValue, ValueId};

const DEFAULT_BLOCK_SIZE: u64 = 100_000;
const DEFAULT_LOCALE: &str = "en_US";

pub struct Slice;

impl TryFrom<TCValue> for Slice {
    type Error = error::TCError;

    fn try_from(_value: TCValue) -> TCResult<Slice> {
        Err(error::not_implemented())
    }
}

pub struct IndexConfig {
    block_size: u64,
    locale: Locale,
    key: HashMap<ValueId, TCPath>,
    values: HashMap<ValueId, TCPath>,
}

impl TryFrom<TCValue> for IndexConfig {
    type Error = error::TCError;

    fn try_from(value: TCValue) -> TCResult<IndexConfig> {
        let mut params: HashMap<String, TCValue> = value.try_into()?;

        let block_size: u64 = params
            .remove("block_size")
            .map(|v| v.try_into())
            .unwrap_or(Ok(DEFAULT_BLOCK_SIZE))?;

        let locale: String = params
            .remove("block_size")
            .map(|v| v.try_into())
            .unwrap_or(Ok(DEFAULT_LOCALE.to_string()))?;
        let locale: Locale = locale.parse()?;

        let key: HashMap<ValueId, TCPath> =
            params
                .remove("key")
                .map(|k| k.try_into())
                .unwrap_or(Err(error::bad_request(
                    "Index key must be specified",
                    TCValue::None,
                )))?;

        let values: HashMap<ValueId, TCPath> = params
            .remove("key")
            .unwrap_or(TCValue::Vector(vec![]))
            .try_into()?;

        Ok(IndexConfig {
            block_size,
            locale,
            key,
            values,
        })
    }
}

pub struct Index {
    config: IndexConfig,
    blocks: Arc<Store>,
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
        // binary-search the list of blocks to find the one that key belongs to
        // read the block
        // binary-search the block to find the correct position for the record
        // insert the record into the block
        // replace the block
        Err(error::not_implemented())
    }
}

#[async_trait]
impl Derived for Index {
    type Config = IndexConfig;

    async fn new(_txn_id: &TxnId, context: Arc<Store>, config: Self::Config) -> TCResult<Self> {
        Ok(Index {
            config,
            blocks: context,
        })
    }
}

impl Extend<PutOp> for Index {
    fn extend<I: IntoIterator<Item = PutOp>>(&mut self, _iter: I) {
        // TODO
    }
}
