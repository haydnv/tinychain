use std::fmt;

use tc_transact::public::helpers::ErrorHandler;
use tc_transact::public::{Handler, Route};
use tcgeneric::PathSegment;

use crate::state::State;

mod cluster;

#[derive(Default)]
pub struct Static {
    state: crate::state::Static,
}

impl Route<State> for Static {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        if path.is_empty() {
            return None;
        }

        if path[0] == tc_state::public::PREFIX {
            self.state.route(&path[1..])
        } else if path[0].as_str() == "error" {
            if path.len() == 2 {
                let code = &path[1];
                Some(Box::new(ErrorHandler::from(code)))
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl fmt::Debug for Static {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("static context")
    }
}
