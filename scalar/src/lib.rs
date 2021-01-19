use generic::{Map, Tuple};
use value::Value;

pub use value::*;

#[derive(Clone)]
pub enum Scalar {
    Map(Map<Self>),
    Tuple(Tuple<Self>),
    Value(Value),
}
