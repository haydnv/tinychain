use tc_transact::public::{Handler, Route};
use tcgeneric::PathSegment;

use crate::cluster::Dir;
use crate::State;

impl<T: Route<State>> Route<State> for Dir<T> {
    fn route<'a>(&'a self, _path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        None
    }
}
