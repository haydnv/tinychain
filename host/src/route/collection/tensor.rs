use tcgeneric::PathSegment;

use crate::collection::Tensor;

use super::{Handler, Route};

impl Route for Tensor {
    fn route<'a>(&'a self, _path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        None
    }
}
