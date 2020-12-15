use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::pin::Pin;

use async_trait::async_trait;
use futures::future::Future;
use futures::stream::Stream;

use crate::chain::{Chain, ChainType};
use crate::collection::{Collection, CollectionType};
use crate::error;
use crate::handler::Public;
use crate::object::{Object, ObjectType};
use crate::request::Request;
use crate::scalar::{
    self, label, Link, PathSegment, Scalar, ScalarType, TCPathBuf, Value, ValueType,
};
use crate::transaction::Txn;

pub type TCBoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a + Send>>;
pub type TCBoxTryFuture<'a, T> = TCBoxFuture<'a, TCResult<T>>;
pub type TCResult<T> = error::TCResult<T>;
pub type TCStream<T> = Pin<Box<dyn Stream<Item = T> + Send + Unpin>>;
pub type TCTryStream<T> = TCStream<TCResult<T>>;

pub trait Class: Into<Link> + Clone + Eq + fmt::Display {
    type Instance: Instance;
}

pub trait NativeClass: Class {
    fn from_path(path: &[PathSegment]) -> TCResult<Self>;

    fn prefix() -> TCPathBuf;
}

pub trait Instance: Clone + Send + Sync {
    type Class: Class;

    fn class(&self) -> Self::Class;

    fn is_a(&self, dtype: Self::Class) -> bool {
        self.class() == dtype
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum TCType {
    Chain(ChainType),
    Collection(CollectionType),
    Object(ObjectType),
    Scalar(ScalarType),
}

impl Class for TCType {
    type Instance = State;
}

impl NativeClass for TCType {
    fn from_path(path: &[PathSegment]) -> TCResult<TCType> {
        let suffix = Self::prefix().try_suffix(path)?;

        match suffix[0].as_str() {
            "chain" => ChainType::from_path(path).map(TCType::Chain),
            "collection" => CollectionType::from_path(path).map(TCType::Collection),
            "object" => ObjectType::from_path(path).map(TCType::Object),
            "op" | "tuple" | "value" => ScalarType::from_path(path).map(TCType::Scalar),
            other => Err(error::not_found(other)),
        }
    }

    fn prefix() -> TCPathBuf {
        label("sbin").into()
    }
}

impl From<ChainType> for TCType {
    fn from(ct: ChainType) -> TCType {
        TCType::Chain(ct)
    }
}

impl From<CollectionType> for TCType {
    fn from(ct: CollectionType) -> TCType {
        TCType::Collection(ct)
    }
}

impl From<ObjectType> for TCType {
    fn from(ot: ObjectType) -> TCType {
        TCType::Object(ot)
    }
}

impl From<ValueType> for TCType {
    fn from(vt: ValueType) -> TCType {
        TCType::Scalar(ScalarType::Value(vt))
    }
}

impl TryFrom<TCType> for ValueType {
    type Error = error::TCError;

    fn try_from(class: TCType) -> TCResult<ValueType> {
        match class {
            TCType::Scalar(ScalarType::Value(class)) => Ok(class),
            other => Err(error::bad_request("Expected ValueType, found", other)),
        }
    }
}

impl From<TCType> for Link {
    fn from(t: TCType) -> Link {
        match t {
            TCType::Chain(ct) => ct.into(),
            TCType::Collection(ct) => ct.into(),
            TCType::Object(ot) => ot.into(),
            TCType::Scalar(st) => st.into(),
        }
    }
}

impl fmt::Display for TCType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Chain(ct) => write!(f, "{}", ct),
            Self::Collection(ct) => write!(f, "{}", ct),
            Self::Object(ot) => write!(f, "{}", ot),
            Self::Scalar(st) => write!(f, "{}", st),
        }
    }
}

#[derive(Clone)]
pub enum State {
    Chain(Chain),
    Collection(Collection),
    Object(Object),
    Scalar(Scalar),
}

impl State {
    pub fn is_none(&self) -> bool {
        match self {
            Self::Scalar(scalar) => scalar.is_none(),
            _ => false,
        }
    }
}

impl Instance for State {
    type Class = TCType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Chain(chain) => chain.class().into(),
            Self::Collection(collection) => collection.class().into(),
            Self::Object(object) => object.class().into(),
            Self::Scalar(scalar) => scalar.class().into(),
        }
    }
}

#[async_trait]
impl Public for State {
    async fn get(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        key: Value,
    ) -> TCResult<State> {
        match self {
            Self::Chain(chain) => chain.get(request, txn, path, key).await,
            Self::Collection(collection) => collection.get(request, txn, path, key).await,
            Self::Object(object) => object.get(request, txn, path, key).await,
            Self::Scalar(scalar) => scalar.get(request, txn, path, key).await,
        }
    }

    async fn put(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        key: Value,
        value: State,
    ) -> TCResult<()> {
        match self {
            Self::Chain(chain) => chain.put(request, txn, path, key, value).await,
            Self::Collection(collection) => collection.put(request, txn, path, key, value).await,
            Self::Object(object) => object.put(request, txn, path, key, value).await,
            Self::Scalar(scalar) => scalar.put(request, txn, path, key, value).await,
        }
    }

    async fn post(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        params: scalar::Object,
    ) -> TCResult<State> {
        match self {
            Self::Chain(chain) => chain.post(request, txn, path, params).await,
            Self::Collection(collection) => collection.post(request, txn, path, params).await,
            Self::Object(object) => object.post(request, txn, path, params).await,
            Self::Scalar(scalar) => scalar.post(request, txn, path, params).await,
        }
    }

    async fn delete(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        key: Value,
    ) -> TCResult<()> {
        match self {
            Self::Chain(chain) => chain.delete(request, txn, path, key).await,
            Self::Collection(collection) => collection.delete(request, txn, path, key).await,
            Self::Object(object) => object.delete(request, txn, path, key).await,
            Self::Scalar(scalar) => scalar.delete(request, txn, path, key).await,
        }
    }
}

impl From<Chain> for State {
    fn from(c: Chain) -> State {
        Self::Chain(c)
    }
}

impl From<Collection> for State {
    fn from(c: Collection) -> State {
        Self::Collection(c)
    }
}

impl From<Object> for State {
    fn from(o: Object) -> State {
        State::Object(o)
    }
}

impl From<Scalar> for State {
    fn from(s: Scalar) -> State {
        Self::Scalar(s)
    }
}

impl From<Value> for State {
    fn from(v: Value) -> State {
        Self::Scalar(Scalar::from(v))
    }
}

impl From<()> for State {
    fn from(_: ()) -> State {
        Self::Scalar(Scalar::Value(Value::None))
    }
}

impl TryFrom<State> for Chain {
    type Error = error::TCError;

    fn try_from(state: State) -> TCResult<Chain> {
        match state {
            State::Chain(chain) => Ok(chain),
            other => Err(error::bad_request("Expected Chain but found", other)),
        }
    }
}

impl TryFrom<State> for Object {
    type Error = error::TCError;

    fn try_from(state: State) -> TCResult<Object> {
        match state {
            State::Object(object) => Ok(object),
            other => Err(error::bad_request("Expected Object but found", other)),
        }
    }
}

impl TryFrom<State> for Scalar {
    type Error = error::TCError;

    fn try_from(state: State) -> TCResult<Scalar> {
        match state {
            State::Scalar(scalar) => Ok(scalar),
            other => Err(error::bad_request("Expected Scalar but found", other)),
        }
    }
}

impl TryFrom<State> for Value {
    type Error = error::TCError;

    fn try_from(state: State) -> TCResult<Value> {
        let scalar = Scalar::try_from(state)?;
        scalar.try_into()
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Chain(c) => write!(f, "{}", c),
            Self::Collection(c) => write!(f, "{}", c),
            Self::Object(o) => write!(f, "{}", o),
            Self::Scalar(s) => write!(f, "{}", s),
        }
    }
}
