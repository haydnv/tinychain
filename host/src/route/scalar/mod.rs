use generic::PathSegment;

use crate::scalar::Scalar;

use super::{Handler, Route};

mod op;
mod value;

impl Route for Scalar {
    fn route<'a>(&'a self, path: &[PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        match self {
            Self::Op(op_def) => op_def.route(path),
            Self::Value(value) => value.route(path),
            _ => None,
        }
    }
}
