use generic::PathSegment;

use crate::scalar::Value;

use super::{Handler, Route};

impl Route for Value {
    fn route<'a>(&'a self, _path: &[PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        None
    }
}
