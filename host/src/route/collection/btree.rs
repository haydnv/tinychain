use tcgeneric::PathSegment;

use crate::collection::BTree;
use crate::route::{Handler, Route};

impl Route for BTree {
    fn route<'a>(&'a self, _path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        todo!()
    }
}
