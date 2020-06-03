use futures::Stream;

use crate::value::link::{Link, TCPath};
use crate::value::{TCRef, Value, ValueId};

pub enum Subject {
    Link(Link),
    TCRef(TCRef),
}

impl From<TCPath> for Subject {
    fn from(path: TCPath) -> Subject {
        Subject::Link(path.into())
    }
}

pub struct Get {
    selector: Value,
}

impl<T: Into<Value>> From<(T,)> for Get {
    fn from(tuple: (T,)) -> Get {
        Get {
            selector: tuple.0.into(),
        }
    }
}

pub struct Put<S: Stream<Item = (Value, Value)>> {
    values: S,
}

impl<S: Stream<Item = (Value, Value)>> From<S> for Put<S> {
    fn from(values: S) -> Put<S> {
        Put { values }
    }
}

pub struct Post<S: Stream<Item = (ValueId, Value)>> {
    data: S,
}
