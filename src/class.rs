use crate::collection::Collection;
use crate::value::Value;

pub struct Object;

pub enum State {
    Collection(Collection),
    Object(Object),
    Value(Value),
}

impl From<Value> for State {
    fn from(v: Value) -> State {
        Self::Value(v)
    }
}
