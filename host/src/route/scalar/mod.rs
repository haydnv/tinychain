use tc_value::{Bound, Range};
use tcgeneric::PathSegment;

use crate::scalar::Scalar;

use super::{Handler, Route};

mod op;
mod value;

impl Route for Range {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            None
        } else {
            match path[0].as_str() {
                "start" => match &self.start {
                    Bound::In(value) => value.route(&path[1..]),
                    Bound::Ex(value) => value.route(&path[1..]),
                    _ => None,
                },
                "end" => match &self.end {
                    Bound::In(value) => value.route(&path[1..]),
                    Bound::Ex(value) => value.route(&path[1..]),
                    _ => None,
                },
                _ => None,
            }
        }
    }
}

impl Route for Scalar {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        match self {
            Self::Map(map) => map.route(path),
            Self::Op(op_def) => op_def.route(path),
            Self::Range(range) => range.route(path),
            Self::Ref(_) => None,
            Self::Value(value) => value.route(path),
            Self::Tuple(tuple) => tuple.route(path),
        }
    }
}
