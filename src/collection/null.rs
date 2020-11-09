use std::fmt;

use async_trait::async_trait;
use futures::stream;

use crate::class::{Class, Instance, NativeClass, TCResult, TCStream};
use crate::collection::class::*;
use crate::collection::{Collection, CollectionBase, CollectionItem};
use crate::error;
use crate::request::Request;
use crate::scalar::{label, Link, PathSegment, Scalar, TCPathBuf, Value};
use crate::transaction::{Transact, Txn, TxnId};

#[derive(Clone, Eq, PartialEq)]
pub struct NullType;

impl Class for NullType {
    type Instance = Null;
}

impl NativeClass for NullType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.is_empty() {
            Ok(NullType)
        } else {
            Err(error::path_not_found(path))
        }
    }

    fn prefix() -> TCPathBuf {
        CollectionType::prefix().append(label("null"))
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

    async fn get(
        &self,
        _request: &Request,
        _txn: &Txn,
        _path: &[PathSegment],
        _selector: Value,
    ) -> TCResult<CollectionItem<Self::Item, Self::Slice>> {
        Err(error::unsupported("Null Collection has no contents to GET"))
    }

    async fn is_empty(&self, _txn: &Txn) -> TCResult<bool> {
        Ok(true)
    }

    async fn put(
        &self,
        _request: &Request,
        _txn: &Txn,
        _path: &[PathSegment],
        _selector: Value,
        _value: CollectionItem<Self::Item, Self::Slice>,
    ) -> TCResult<()> {
        Err(error::unsupported("Null Collection cannot be modified"))
    }

    async fn to_stream(&self, _txn: Txn) -> TCResult<TCStream<Scalar>> {
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

    async fn finalize(&self, _txn_id: &TxnId) {
        // no-op
    }
}

impl From<Null> for Collection {
    fn from(null: Null) -> Collection {
        Collection::Base(CollectionBase::Null(null))
    }
}
