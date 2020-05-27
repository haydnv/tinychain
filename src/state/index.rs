use std::collections::HashMap;
use std::convert::TryFrom;
use std::sync::Arc;

use async_trait::async_trait;

use crate::auth::Token;
use crate::error;
use crate::internal::block::Store;
use crate::state::{Collection, Derived, State, Tensor};
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

struct List {
    blocks: Arc<Store>,
}

enum Column {
    List(List),
    Tensor(Tensor),
}

pub struct Index {
    key: HashMap<ValueId, Column>,
    values: HashMap<ValueId, Column>,
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

    async fn get(
        self: &Arc<Self>,
        _txn: &Arc<Txn<'_>>,
        _key: &Slice,
        _auth: &Option<Token>,
    ) -> TCResult<Index> {
        Err(error::not_implemented())
    }

    async fn put(
        self: Arc<Self>,
        _txn: &Arc<Txn<'_>>,
        _key: Slice,
        _value: Index,
        _auth: &Option<Token>,
    ) -> TCResult<State> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl Derived for Index {
    type Config = (Vec<(ValueId, TCPath)>, Vec<(ValueId, TCPath)>);

    async fn create(_txn: &Arc<Txn<'_>>, _config: Self::Config) -> TCResult<Arc<Self>> {
        Err(error::not_implemented())
    }
}
