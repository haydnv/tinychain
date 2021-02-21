use std::iter;

use futures::future;

use tcgeneric::PathSegment;

use crate::scalar::op::*;
use crate::scalar::{Scalar, Value};
use crate::state::State;
use crate::txn::*;

use crate::route::*;

struct OpHandler<'a> {
    op_def: &'a OpDef,
}

impl<'a> Handler<'a> for OpHandler<'a> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        let handle: GetHandler<'a> = match self.op_def {
            OpDef::Get((_, op_def)) if op_def.is_empty() => {
                let handle = |_: Txn, _: Value| {
                    let result: GetFuture<'a> = Box::pin(future::ready(Ok(State::default())));
                    result
                };

                Box::new(handle)
            }
            OpDef::Get(get_op) => {
                let (key_name, op_def) = get_op.clone();

                let handle = move |txn: Txn, key: Value| {
                    let capture = op_def.last().unwrap().0.clone();
                    let op_def =
                        iter::once((key_name, Scalar::Value(key))).chain(op_def.into_iter());
                    let executor = Executor::new(txn, State::default(), op_def);
                    let result: GetFuture<'a> = Box::pin(executor.capture(capture));
                    result
                };

                Box::new(handle)
            }
            _ => unimplemented!(),
        };

        Some(handle)
    }
}

impl Route for OpDef {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            Some(Box::new(OpHandler { op_def: self }))
        } else {
            None
        }
    }
}
