use crate::value::class::{Impl, NumberClass};
use crate::value::Number;

pub fn and(left: Option<Number>, right: Option<Number>) -> Option<Number> {
    match (left, right) {
        (Some(l), Some(r)) if l == l.class().zero() || r == r.class().zero() => None,
        (Some(_), Some(_)) => Some(Number::Bool(true)),
        _ => None,
    }
}
