use generic::PathSegment;

use crate::chain::Chain;

use super::{Handler, Route};

impl Route for Chain {
    fn route<'a>(&'a self, _path: &[PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        None
    }
}
