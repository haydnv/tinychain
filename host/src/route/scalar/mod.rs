use tcgeneric::PathSegment;

use crate::scalar::Scalar;

use super::{Handler, Route};

mod op;
mod value;

impl Route for Scalar {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        match self {
            Self::Map(map) => map.route(path),
            Self::Op(op_def) => op_def.route(path),
            Self::Ref(_) => None,
            Self::Value(value) => value.route(path),
            Self::Tuple(tuple) => tuple.route(path),
        }
    }
}
