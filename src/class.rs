use std::convert::TryFrom;
use std::fmt;
use std::pin::Pin;

use futures::future::Future;
use futures::stream::Stream;

use crate::collection::{Collection, CollectionType};
use crate::error;
use crate::value::link::{Link, TCPath};
use crate::value::number::NumberType;
use crate::value::{label, Value, ValueId, ValueType};

const ERR_EMPTY_CLASSPATH: &str = "Expected a class path, \
e.g. /sbin/value/number/int/64 or /sbin/collection/table, but found: ";

pub const ERR_PROTECTED: &str =
    "You have accessed a protected class. This should not be possible. \
Please file a bug report.";

pub type ResponseStream = TCStream<(ValueId, TCStream<Value>)>;
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
    Collection(CollectionType),
    Value(ValueType),
}

impl Class for TCType {
    type Instance = State;

    fn from_path(path: &TCPath) -> TCResult<TCType> {
        let path = path.from_path(&Self::prefix())?;

        if path.len() > 1 {
            match path[1].as_str() {
                "collection" => {
                    CollectionType::from_path(&path.slice_from(2)).map(TCType::Collection)
                }
                "value" => ValueType::from_path(&path.slice_from(2)).map(TCType::Value),
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
            TCType::Collection(ct) => ct.into(),
            TCType::Value(vt) => vt.into(),
        }
    }
}

impl fmt::Display for TCType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Collection(ctype) => write!(f, "{}", ctype),
            Self::Value(vtype) => write!(f, "{}", vtype),
        }
    }
}

#[derive(Clone)]
pub enum State {
    Collection(Collection),
    Value(Value),
}

impl Instance for State {
    type Class = TCType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Collection(collection) => collection.class().into(),
            Self::Value(value) => value.class().into(),
        }
    }
}

impl From<Value> for State {
    fn from(v: Value) -> State {
        Self::Value(v)
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
            Self::Collection(c) => write!(f, "{}", c),
            Self::Value(v) => write!(f, "{}", v),
        }
    }
}
