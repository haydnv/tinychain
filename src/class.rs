use std::convert::TryFrom;
use std::fmt;
use std::pin::Pin;

use futures::future::Future;
use futures::stream::Stream;

use crate::chain::{Chain, ChainType};
use crate::cluster::Cluster;
use crate::collection::{Collection, CollectionType};
use crate::error;
use crate::value::link::{Link, TCPath};
use crate::value::number::NumberType;
use crate::value::{label, Value, ValueType};

const ERR_EMPTY_CLASSPATH: &str = "Expected a class path, \
e.g. /sbin/value/number/int/64 or /sbin/collection/table, but found: ";

pub const ERR_PROTECTED: &str =
    "You have accessed a protected class. This should not be possible. \
Please file a bug report.";

pub type TCBoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a + Send + Sync>>;
pub type TCBoxTryFuture<'a, T> = TCBoxFuture<'a, TCResult<T>>;
pub type TCResult<T> = Result<T, error::TCError>;
pub type TCStream<T> = Pin<Box<dyn Stream<Item = T> + Send + Sync + Unpin>>;
pub type TCTryStream<T> = TCStream<TCResult<T>>;

pub trait Class: Into<Link> + Clone + Eq + fmt::Display {
    type Instance: Instance;

    fn from_path(path: &TCPath) -> TCResult<Self>;

    fn prefix() -> TCPath;
}

pub trait Instance {
    type Class: Class;

    fn class(&self) -> Self::Class;

    fn is_a(&self, dtype: Self::Class) -> bool {
        self.class() == dtype
    }

    fn expect<M: fmt::Display>(&self, dtype: Self::Class, context_msg: M) -> TCResult<()> {
        if self.is_a(dtype.clone()) {
            Ok(())
        } else {
            Err(error::TCError::of(
                error::Code::BadRequest,
                format!(
                    "Expected {} but found {} {}",
                    self.class(),
                    dtype,
                    context_msg
                ),
            ))
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum TCType {
    Chain(ChainType),
    Cluster,
    Collection(CollectionType),
    Value(ValueType),
}

impl Class for TCType {
    type Instance = State;

    fn from_path(path: &TCPath) -> TCResult<TCType> {
        let suffix = path.from_path(&Self::prefix())?;

        if suffix.len() > 1 {
            match suffix[0].as_str() {
                "chain" => ChainType::from_path(path).map(TCType::Chain),
                "collection" => CollectionType::from_path(path).map(TCType::Collection),
                "value" => ValueType::from_path(path).map(TCType::Value),
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::bad_request(ERR_EMPTY_CLASSPATH, path))
        }
    }

    fn prefix() -> TCPath {
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

impl From<ValueType> for TCType {
    fn from(vt: ValueType) -> TCType {
        TCType::Value(vt)
    }
}

impl TryFrom<TCType> for NumberType {
    type Error = error::TCError;

    fn try_from(class: TCType) -> TCResult<NumberType> {
        match class {
            TCType::Value(ValueType::Number(nt)) => Ok(nt),
            other => Err(error::bad_request("Expected ValueType, found", other)),
        }
    }
}

impl TryFrom<TCType> for ValueType {
    type Error = error::TCError;

    fn try_from(class: TCType) -> TCResult<ValueType> {
        match class {
            TCType::Value(class) => Ok(class),
            other => Err(error::bad_request("Expected ValueType, found", other)),
        }
    }
}

impl From<TCType> for Link {
    fn from(t: TCType) -> Link {
        match t {
            TCType::Chain(ct) => ct.into(),
            TCType::Cluster => TCType::prefix().join(label("cluster").into()).into(),
            TCType::Collection(ct) => ct.into(),
            TCType::Value(vt) => vt.into(),
        }
    }
}

impl fmt::Display for TCType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Chain(ct) => write!(f, "{}", ct),
            Self::Cluster => write!(f, "type Cluster"),
            Self::Collection(ct) => write!(f, "{}", ct),
            Self::Value(vt) => write!(f, "{}", vt),
        }
    }
}

#[derive(Clone)]
pub enum State {
    Chain(Chain),
    Cluster(Cluster),
    Collection(Collection),
    Value(Value),
}

impl Instance for State {
    type Class = TCType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Chain(chain) => chain.class().into(),
            Self::Cluster(_) => TCType::Cluster,
            Self::Collection(collection) => collection.class().into(),
            Self::Value(value) => value.class().into(),
        }
    }
}

impl From<Chain> for State {
    fn from(c: Chain) -> State {
        Self::Chain(c)
    }
}

impl From<Cluster> for State {
    fn from(c: Cluster) -> State {
        Self::Cluster(c)
    }
}

impl From<Collection> for State {
    fn from(c: Collection) -> State {
        Self::Collection(c)
    }
}

impl From<Value> for State {
    fn from(v: Value) -> State {
        Self::Value(v)
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

impl TryFrom<State> for Value {
    type Error = error::TCError;

    fn try_from(state: State) -> TCResult<Value> {
        match state {
            State::Value(value) => Ok(value),
            other => Err(error::bad_request("Expected Value but found", other)),
        }
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Chain(c) => write!(f, "{}", c),
            Self::Cluster(c) => write!(f, "{}", c),
            Self::Collection(c) => write!(f, "{}", c),
            Self::Value(v) => write!(f, "{}", v),
        }
    }
}
