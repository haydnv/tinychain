use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::try_join_all;

use crate::auth::Token;
use crate::error;
use crate::internal::block::Store;
use crate::internal::file::*;
use crate::state::{Collection, Persistent, State};
use crate::transaction::{Txn, TxnId};
use crate::value::link::TCPath;
use crate::value::{TCResult, TCValue};

const BLOCK_SIZE: usize = 1_000_000;

pub struct Slice;

impl TryFrom<TCValue> for Slice {
    type Error = error::TCError;

    fn try_from(_value: TCValue) -> TCResult<Slice> {
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

impl TryFrom<TCValue> for TensorConfig {
    type Error = error::TCError;

    fn try_from(value: TCValue) -> TCResult<TensorConfig> {
        let (data_type, dims): (TCPath, Vec<u64>) = value.try_into()?;
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

impl TryFrom<TCValue> for Tensor {
    type Error = error::TCError;

    fn try_from(_value: TCValue) -> TCResult<Tensor> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl Collection for Tensor {
    type Key = Slice;
    type Value = Tensor;

    async fn get(
        self: &Arc<Self>,
        _txn: &Arc<Txn<'_>>,
        _key: &Slice,
        _auth: &Option<Token>,
    ) -> TCResult<Tensor> {
        Err(error::not_implemented())
    }

    async fn put(
        self: Arc<Self>,
        _txn: &Arc<Txn<'_>>,
        _key: Slice,
        _value: Tensor,
        _auth: &Option<Token>,
    ) -> TCResult<State> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl File for Tensor {
    async fn copy_from(_reader: &mut FileCopier, _txn_id: &TxnId, _dest: Arc<Store>) -> Arc<Self> {
        // TODO
        panic!("Tensor::copy_from is not implemented")
    }

    async fn copy_into(&self, _txn_id: TxnId, _writer: &mut FileCopier) {
        // TODO
        panic!("Tensor::copy_into is not implemented")
    }

    async fn from_store(_txn_id: &TxnId, _store: Arc<Store>) -> Arc<Tensor> {
        // TODO
        panic!("Tensor::from_store is not implemented")
    }
}

#[async_trait]
impl Persistent for Tensor {
    type Config = TensorConfig;

    async fn create(txn: &Arc<Txn<'_>>, config: TensorConfig) -> TCResult<Arc<Self>> {
        let size =
            (config.data_type.size() as u64) * config.dims.iter().fold(1, |size, d| size * d);
        let blocks = txn.context();
        let txn_id = txn.id();

        let num_blocks: usize = (size as f64 / BLOCK_SIZE as f64).ceil() as usize;
        let mut new_blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks + 1 {
            new_blocks.push(blocks.new_block(&txn_id, i.into(), Bytes::from(&[0; BLOCK_SIZE][..])))
        }
        try_join_all(new_blocks).await?;

        Ok(Arc::new(Tensor { config, blocks }))
    }
}
