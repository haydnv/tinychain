use tc_transact::public::{Handler, Route};
use tcgeneric::PathSegment;

use crate::cluster::Service;
use crate::State;

impl Route<State> for Service {
    fn route<'a>(&'a self, _path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        todo!()
    }
}
