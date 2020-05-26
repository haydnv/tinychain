use std::convert::TryFrom;
use std::sync::Arc;

use async_trait::async_trait;

use crate::auth::Token;
use crate::error;
use crate::internal::block::Store;
use crate::internal::file::*;
use crate::state::{Collection, State};
use crate::transaction::{Txn, TxnId};
use crate::value::{TCResult, TCValue};

pub struct Slice;

impl TryFrom<TCValue> for Slice {
    type Error = error::TCError;

    fn try_from(_value: TCValue) -> TCResult<Slice> {
        Err(error::not_implemented())
    }
}

pub struct Tensor;

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
        Arc::new(Tensor)
    }

    async fn copy_into(&self, _txn_id: TxnId, _writer: &mut FileCopier) {
        // TODO
    }

    async fn from_store(_txn_id: &TxnId, _store: Arc<Store>) -> Arc<Tensor> {
        // TODO
        Arc::new(Tensor)
    }
}
