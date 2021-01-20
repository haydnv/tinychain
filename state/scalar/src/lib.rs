use generic::{Map, Tuple};

pub use value::*;

#[derive(Clone)]
pub enum Scalar {
    Map(Map<Self>),
    Tuple(Tuple<Self>),
    Value(Value),
}
