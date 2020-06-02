use futures::Stream;

use crate::value::link::Link;
use crate::value::{TCRef, Value, ValueId};

enum Subject {
    Link(Link),
    TCRef(TCRef),
}

pub struct Get {
    selector: Value,
}

pub struct Put<S: Stream<Item = Value>> {
    selector: Value,
    values: S,
}

pub struct Post<S: Stream<Item = (ValueId, Value)>> {
    data: S,
}
