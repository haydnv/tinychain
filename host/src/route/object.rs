use std::collections::HashMap;

use log::debug;

use error::*;
use generic::{Id, Instance, Map, PathSegment, TCPath};

use crate::object::InstanceExt;
use crate::scalar::*;
use crate::state::State;
use crate::txn::Txn;

use super::{GetHandler, Handler, PostHandler, PutHandler, Route};

struct GetMethod<'a> {
    subject: State,
    method: GetOp,
    path: &'a [PathSegment],
}

impl<'a> GetMethod<'a> {
    async fn call(self, txn: Txn, key: Value) -> TCResult<State> {
        let (key_name, op_def) = self.method;

        let mut context = HashMap::with_capacity(2);
        context.insert(SELF.into(), self.subject);
        context.insert(key_name, key.into());

        call_method(txn, self.path, context.into(), op_def).await
    }
}

impl<'a> Handler<'a> for GetMethod<'a> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(move |txn, key| Box::pin(self.call(txn, key))))
    }
}

struct PutMethod<'a> {
    subject: State,
    method: PutOp,
    path: &'a [PathSegment],
}

impl<'a> PutMethod<'a> {
    async fn call(self, txn: Txn, key: Value, value: State) -> TCResult<()> {
        let (key_name, value_name, op_def) = self.method;

        let mut context = HashMap::with_capacity(3);
        context.insert(SELF.into(), self.subject);
        context.insert(key_name, key.into());
        context.insert(value_name, value);

        let state = call_method(txn, self.path, context.into(), op_def).await?;
        if state.is_none() {
            Ok(())
        } else {
            Err(TCError::bad_request(
                "PUT method should return None, not",
                state,
            ))
        }
    }
}

impl<'a> Handler<'a> for PutMethod<'a> {
    fn put(self: Box<Self>) -> Option<PutHandler<'a>> {
        Some(Box::new(move |txn, key, value| {
            Box::pin(self.call(txn, key, value))
        }))
    }
}

struct PostMethod<'a> {
    subject: State,
    method: PostOp,
    path: &'a [PathSegment],
}

impl<'a> PostMethod<'a> {
    async fn call(self, txn: Txn, params: Map<State>) -> TCResult<State> {
        let mut context = HashMap::with_capacity(params.len() + 1);
        context.insert(SELF.into(), self.subject);
        context.extend(params);

        call_method(txn, self.path, context.into(), self.method).await
    }
}

impl<'a> Handler<'a> for PostMethod<'a> {
    fn post(self: Box<Self>) -> Option<PostHandler<'a>> {
        Some(Box::new(move |txn, params| {
            Box::pin(self.call(txn, params))
        }))
    }
}

impl<T: Instance + Route + Clone> Route for InstanceExt<T>
where
    State: From<T>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        debug!("InstanceExt::route {}", TCPath::from(path));

        if path.is_empty() {
            None
        } else if let Some(member) = self.proto().get(&path[0]) {
            match member {
                Scalar::Op(OpDef::Get(get_op)) => Some(Box::new(GetMethod {
                    subject: self.clone().into(),
                    method: get_op.clone(),
                    path,
                })),
                Scalar::Op(OpDef::Put(put_op)) => Some(Box::new(PutMethod {
                    subject: self.clone().into(),
                    method: put_op.clone(),
                    path,
                })),
                Scalar::Op(OpDef::Post(post_op)) => Some(Box::new(PostMethod {
                    subject: self.clone().into(),
                    method: post_op.clone(),
                    path,
                })),
                other => other.route(&path[1..]),
            }
        } else {
            self.parent().route(path)
        }
    }
}

async fn call_method(
    txn: Txn,
    path: &[PathSegment],
    context: Map<State>,
    form: Vec<(Id, Scalar)>,
) -> TCResult<State> {
    if !path.is_empty() {
        return Err(TCError::not_found(TCPath::from(path)));
    }

    let capture = if let Some((capture, _)) = form.last() {
        capture.clone()
    } else {
        return Ok(State::default());
    };

    Executor::with_context(txn, context.into(), form)
        .capture(capture)
        .await
}
