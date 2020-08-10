use std::collections::HashMap;
use std::fmt;

use crate::collection::{Collection, CollectionType};
use crate::error;
use crate::value::{TCResult, Value, ValueId, ValueType};

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
