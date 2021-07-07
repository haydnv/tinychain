use std::fmt;

use crate::generic::PathSegment;

#[cfg(feature = "tensor")]
use super::collection::{EinsumHandler, EINSUM};
use super::{Handler, Route};

pub struct Static;

impl Route for Static {
    #[allow(unused_variables)]
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        #[cfg(feature = "tensor")]
        if path == &EINSUM[..] {
            return Some(Box::new(EinsumHandler));
        }

        None
    }
}

impl fmt::Display for Static {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("static context")
    }
}
