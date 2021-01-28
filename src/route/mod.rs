use std::pin::Pin;

use futures::Future;

use error::*;
use generic::PathSegment;
use value::Value;

use crate::state::State;
use crate::txn::Txn;

mod state;

pub type GetHandler<'a> =
    Box<dyn FnOnce(&Txn, Value) -> Pin<Box<dyn Future<Output = TCResult<State>> + 'a>> + 'a>;

pub trait Handler<'a> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        None
    }
}

pub trait Route {
    fn route<'a>(&'a self, path: &[PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>>;
}
