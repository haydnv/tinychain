use tc_table::Table;

use super::{Handler, Route};
use crate::generic::PathSegment;

impl Route for Table {
    fn route<'a>(&'a self, _path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        // TODO
        None
    }
}
