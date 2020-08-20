use std::collections::HashMap;
use std::fmt;
use std::pin::Pin;

use futures::future::Future;
use futures::stream::Stream;

use crate::collection::{Collection, CollectionType};
use crate::error;
use crate::value::{Value, ValueId, ValueType};

pub type TCBoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a + Send + Sync>>;
pub type TCBoxTryFuture<'a, T> = TCBoxFuture<'a, TCResult<T>>;
pub type TCResult<T> = Result<T, error::TCError>;
pub type TCStream<T> = Pin<Box<dyn Stream<Item = T> + Send + Sync + Unpin>>;
pub type TCTryStream<T> = TCStream<TCResult<T>>;

pub trait Class: Clone + Eq + fmt::Display {
    type Instance: Instance;
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

pub enum TCType {
    Collection(CollectionType),
    Value(ValueType),
}

pub enum ClassMember {
    Type(TCType),
    Class(Box<ClassMember>),
}

pub type ClassDef = HashMap<ValueId, ClassMember>;

#[derive(Clone)]
pub enum State {
    Class(),
    Collection(Collection),
    Value(Value),
}

impl From<Value> for State {
    fn from(v: Value) -> State {
        Self::Value(v)
    }
}
