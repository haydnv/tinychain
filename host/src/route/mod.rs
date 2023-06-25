use std::fmt;

use tc_error::*;
use tc_transact::public::helpers::ErrorHandler;
use tc_transact::public::{Handler, Route};
use tcgeneric::{Id, PathSegment};

use crate::state::State;
use crate::txn::Txn;

mod chain;
mod cluster;
mod object;
mod state;
// mod stream;

pub type GetFuture<'a> = tc_transact::public::GetFuture<'a, State>;
pub type GetHandler<'a, 'b> = tc_transact::public::GetHandler<'a, 'b, Txn, State>;

pub type PutFuture<'a> = tc_transact::public::PutFuture<'a>;
pub type PutHandler<'a, 'b> = tc_transact::public::PutHandler<'a, 'b, Txn, State>;

pub type PostFuture<'a> = tc_transact::public::PostFuture<'a, State>;
pub type PostHandler<'a, 'b> = tc_transact::public::PostHandler<'a, 'b, Txn, State>;

pub type DeleteFuture<'a> = tc_transact::public::DeleteFuture<'a>;
pub type DeleteHandler<'a, 'b> = tc_transact::public::DeleteHandler<'a, 'b, Txn>;

pub struct Static;

impl Route<State> for Static {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        if path.is_empty() {
            return None;
        }

        if path[0] == state::PREFIX {
            state::Static.route(&path[1..])
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

fn error_type(err_type: &Id) -> Option<ErrorKind> {
    match err_type.as_str() {
        "bad_gateway" => Some(ErrorKind::BadGateway),
        "bad_request" => Some(ErrorKind::BadRequest),
        "conflict" => Some(ErrorKind::Conflict),
        "forbidden" => Some(ErrorKind::Forbidden),
        "internal" => Some(ErrorKind::Internal),
        "method_not_allowed" => Some(ErrorKind::MethodNotAllowed),
        "not_found" => Some(ErrorKind::NotFound),
        "not_implemented" => Some(ErrorKind::NotImplemented),
        "timeout" => Some(ErrorKind::Timeout),
        "unauthorized" => Some(ErrorKind::Unauthorized),
        _ => None,
    }
}
