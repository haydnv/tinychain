use std::pin::Pin;

use futures::Future;

use error::*;
use generic::PathSegment;
use value::Value;

use crate::state::State;
use crate::txn::Txn;

mod op_def;
mod state;

pub type GetFuture<'a> = Pin<Box<dyn Future<Output = TCResult<State>> + 'a>>;
pub type GetHandler<'a> = Box<dyn FnOnce(&'a Txn, Value) -> GetFuture + 'a>;

pub trait Handler<'a> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        None
    }
}

pub trait Route {
    fn route<'a>(&'a self, path: &[PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>>;
}
