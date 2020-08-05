use crate::value::class::{Impl, NumberClass};
use crate::value::Number;

pub fn and(left: Number, right: Number) -> Number {
    if left == left.class().zero() || right == right.class().zero() {
        Number::Bool(false)
    } else {
        Number::Bool(true)
    }
}
