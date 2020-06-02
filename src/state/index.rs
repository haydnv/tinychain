use std::collections::HashMap;
use std::convert::TryFrom;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error;
use crate::i18n::Locale;
use crate::internal::Store;
use crate::state::{Args, Collection, Derived, State};
use crate::transaction::{Txn, TxnId};
use crate::value::link::TCPath;
use crate::value::op::PutOp;
use crate::value::{TCResult, Value, ValueId};
use crate::DEFAULT_LOCALE;

const DEFAULT_BLOCK_SIZE: u64 = 100_000;

pub struct Slice;

impl TryFrom<Value> for Slice {
    type Error = error::TCError;

    fn try_from(_value: Value) -> TCResult<Slice> {
        Err(error::not_implemented())
    }
}

pub struct IndexConfig {
    block_size: u64,
    locale: Locale,
    key: HashMap<ValueId, TCPath>,
    values: HashMap<ValueId, TCPath>,
}

impl TryFrom<Args> for IndexConfig {
    type Error = error::TCError;

    fn try_from(mut args: Args) -> TCResult<IndexConfig> {
        let block_size: u64 = args.take_or("block_size", DEFAULT_BLOCK_SIZE)?;
        let locale: String = args.take_or("locale", DEFAULT_LOCALE.to_string())?;
        let locale: Locale = locale.parse()?;
        let key: HashMap<ValueId, TCPath> = args.take("key")?;
        let values: HashMap<ValueId, TCPath> = args.take("values")?;

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

impl TryFrom<Value> for Index {
    type Error = error::TCError;

    fn try_from(_value: Value) -> TCResult<Index> {
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
