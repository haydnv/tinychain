use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream;

use crate::class::{Class, Instance, TCResult, TCStream};
use crate::collection::class::*;
use crate::collection::{Collection, CollectionBase, CollectionItem};
use crate::error;
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::{label, Link, TCPath, Value};

#[derive(Clone, Eq, PartialEq)]
pub struct NullType;

impl Class for NullType {
    type Instance = Null;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        let suffix = path.from_path(&Self::prefix())?;

        if suffix.is_empty() {
            Ok(NullType)
        } else {
            Err(error::not_found(path))
        }
    }

    fn prefix() -> TCPath {
        CollectionType::prefix().join(label("null").into())
    }
}

impl From<NullType> for CollectionType {
    fn from(_: NullType) -> CollectionType {
        CollectionType::Base(CollectionBaseType::Null)
    }
}

impl From<NullType> for Link {
    fn from(_: NullType) -> Link {
        NullType::prefix().into()
    }
}

impl fmt::Display for NullType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "type: Null Collection")
    }
}

#[derive(Clone)]
pub struct Null;

impl Null {
    pub fn create() -> Null {
        Null
    }
}

impl Instance for Null {
    type Class = NullType;

    fn class(&self) -> NullType {
        NullType
    }
}

#[async_trait]
impl CollectionInstance for Null {
    type Item = Value;
    type Slice = Null;

    async fn get_item(
        &self,
        _txn: Arc<Txn>,
        _selector: Value,
    ) -> TCResult<CollectionItem<Self::Item, Self::Slice>> {
        Err(error::unsupported("Null Collection has no contents to GET"))
    }

    async fn is_empty(&self, _txn: Arc<Txn>) -> TCResult<bool> {
        Ok(true)
    }

    async fn put_item(
        &self,
        _txn: Arc<Txn>,
        _selector: Value,
        _value: CollectionItem<Self::Item, Self::Slice>,
    ) -> TCResult<()> {
        Err(error::unsupported("Null Collection cannot be modified"))
    }

    async fn to_stream(&self, _txn: Arc<Txn>) -> TCResult<TCStream<Value>> {
        Ok(Box::pin(stream::empty()))
    }
}

#[async_trait]
impl Transact for Null {
    async fn commit(&self, _txn_id: &TxnId) {
        // no-op
    }

    async fn rollback(&self, _txn_id: &TxnId) {
        // no-op
    }
}

impl From<Null> for Collection {
    fn from(null: Null) -> Collection {
        Collection::Base(CollectionBase::Null(null))
    }
}
