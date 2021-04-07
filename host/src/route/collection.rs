use tcgeneric::PathSegment;

use crate::collection::Collection;

use super::{Handler, Route};

impl Route for Collection {
    fn route<'a>(&'a self, _path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        todo!()
    }
}
