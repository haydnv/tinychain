use std::fmt;
use std::sync::Arc;

use futures::{future, stream};

use crate::class::{Class, Instance, TCBoxFuture, TCBoxTryFuture, TCResult, TCStream};
use crate::collection::class::*;
use crate::collection::{Collection, CollectionBase, CollectionItem};
use crate::error;
use crate::scalar::{label, Link, Scalar, TCPath, Value};
use crate::transaction::{Transact, Txn, TxnId};

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

impl CollectionInstance for Null {
    type Item = Value;
    type Slice = Null;

    fn get<'a>(
        &'a self,
        _txn: Arc<Txn>,
        _path: TCPath,
        _selector: Value,
    ) -> TCBoxTryFuture<'a, CollectionItem<Self::Item, Self::Slice>> {
        Box::pin(future::ready(Err(error::unsupported(
            "Null Collection has no contents to GET",
        ))))
    }

    fn is_empty<'a>(&'a self, _txn: Arc<Txn>) -> TCBoxTryFuture<'a, bool> {
        Box::pin(future::ready(Ok(true)))
    }

    fn put<'a>(
        &'a self,
        _txn: Arc<Txn>,
        _path: TCPath,
        _selector: Value,
        _value: CollectionItem<Self::Item, Self::Slice>,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(future::ready(Err(error::unsupported(
            "Null Collection cannot be modified",
        ))))
    }

    fn to_stream<'a>(&'a self, _txn: Arc<Txn>) -> TCBoxTryFuture<'a, TCStream<Scalar>> {
        Box::pin(future::ready({
            let stream: TCStream<Scalar> = Box::pin(stream::empty());
            Ok(stream)
        }))
    }
}

impl Transact for Null {
    fn commit<'a>(&'a self, _txn_id: &'a TxnId) -> TCBoxFuture<'a, ()> {
        // no-op
        Box::pin(future::ready(()))
    }

    fn rollback<'a>(&'a self, _txn_id: &'a TxnId) -> TCBoxFuture<'a, ()> {
        // no-op
        Box::pin(future::ready(()))
    }
}

impl From<Null> for Collection {
    fn from(null: Null) -> Collection {
        Collection::Base(CollectionBase::Null(null))
    }
}
