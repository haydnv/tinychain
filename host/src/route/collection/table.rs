use tcgeneric::PathSegment;

use crate::collection::Table;

use super::{Handler, Route};

impl Route for Table {
    fn route<'a>(&'a self, _path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        // TODO
        None
    }
}
