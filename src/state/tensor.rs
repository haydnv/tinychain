use std::convert::TryFrom;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::try_join_all;

use crate::error;
use crate::internal::Store;
use crate::state::{Args, Collection, Derived, State};
use crate::transaction::{Txn, TxnId};
use crate::value::link::TCPath;
use crate::value::op::PutOp;
use crate::value::{TCResult, Value};

const BLOCK_SIZE: usize = 1_000_000;

pub struct Slice;

impl TryFrom<Value> for Slice {
    type Error = error::TCError;

    fn try_from(_value: Value) -> TCResult<Slice> {
        Err(error::not_implemented())
    }
}

enum DataType {
    Int32,
    UInt64,
}

impl DataType {
    fn size(&self) -> u8 {
        use DataType::*;
        match self {
            Int32 => 32,
            UInt64 => 64,
        }
    }
}

pub struct TensorConfig {
    data_type: DataType,
    dims: Vec<u64>,
}

impl TryFrom<Args> for TensorConfig {
    type Error = error::TCError;

    fn try_from(mut args: Args) -> TCResult<TensorConfig> {
        let data_type: TCPath = args.take("data_type")?;
        let dims: Vec<u64> = args.take("dims")?;

        if !data_type.starts_with("/sbin/value/number".parse().unwrap()) || data_type.len() < 4 {
            return Err(error::bad_request(
                "Tensor data type must be numeric, found",
                data_type,
            ));
        }

        let data_type = match data_type.slice_from(4).to_string().as_str() {
            "/i32" => DataType::Int32,
            "/u64" => DataType::UInt64,
            other => {
                return Err(error::bad_request(
                    "Not a supported tensor data type",
                    other,
                ));
            }
        };

        Ok(TensorConfig { data_type, dims })
    }
}

pub struct Tensor {
    config: TensorConfig,
    blocks: Arc<Store>,
}

impl TryFrom<Value> for Tensor {
    type Error = error::TCError;

    fn try_from(_value: Value) -> TCResult<Tensor> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl Collection for Tensor {
    type Key = Slice;
    type Value = Tensor;

    async fn get(self: &Arc<Self>, _txn: &Arc<Txn<'_>>, _key: &Slice) -> TCResult<Tensor> {
        Err(error::not_implemented())
    }

    async fn put(
        self: Arc<Self>,
        _txn: &Arc<Txn<'_>>,
        _key: Slice,
        _value: Tensor,
    ) -> TCResult<State> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl Derived for Tensor {
    type Config = TensorConfig;

    async fn new(txn_id: &TxnId, context: Arc<Store>, config: TensorConfig) -> TCResult<Self> {
        let size: u64 = (config.data_type.size() as u64) * config.dims.iter().product::<u64>();
        let num_blocks: usize = (size as f64 / BLOCK_SIZE as f64).ceil() as usize;

        let mut new_blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks + 1 {
            new_blocks.push(context.new_block(
                txn_id.clone(),
                i.into(),
                Bytes::from(&[0; BLOCK_SIZE][..]),
            ))
        }
        try_join_all(new_blocks).await?;

        Ok(Tensor {
            config,
            blocks: context,
        })
    }
}

impl Extend<PutOp> for Tensor {
    fn extend<I: IntoIterator<Item = PutOp>>(&mut self, _iter: I) {
        // TODO
    }
}
