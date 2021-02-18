use generic::PathSegment;

use crate::scalar::Value;

use super::{Handler, Route};

mod number;

impl Route for Value {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        match self {
            Self::Number(number) => number.route(path),
            _ => None,
        }
    }
}
