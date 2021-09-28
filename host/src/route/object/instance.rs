use std::fmt;

use log::debug;

use tc_error::*;
use tc_value::Value;
use tcgeneric::{Id, Instance, Map, PathSegment, TCPath};

use crate::object::InstanceExt;
use crate::route::{DeleteHandler, GetHandler, Handler, PostHandler, PutHandler, Route};
use crate::scalar::*;
use crate::state::{State, ToState};
use crate::txn::Txn;

struct GetMethod<'a, T: Instance> {
    subject: &'a InstanceExt<T>,
    method: GetOp,
    path: &'a [PathSegment],
}

impl<'a, T: Instance + Route + fmt::Display + 'a> GetMethod<'a, T>
where
    InstanceExt<T>: ToState,
{
    async fn call(self, txn: &Txn, key: Value) -> TCResult<State> {
        let (key_name, op_def) = self.method;

        let mut context = Map::new();
        context.insert(key_name, key.into());

        call_method(txn, self.subject, self.path, context, op_def).await
    }
}

impl<'a, T: Instance + Route + fmt::Display + 'a> Handler<'a> for GetMethod<'a, T>
where
    InstanceExt<T>: ToState,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(move |txn, key| Box::pin(self.call(txn, key))))
    }
}

struct PutMethod<'a, T: Instance> {
    subject: &'a InstanceExt<T>,
    method: PutOp,
    path: &'a [PathSegment],
}

impl<'a, T: Instance + Route + fmt::Display + 'a> PutMethod<'a, T>
where
    InstanceExt<T>: ToState,
{
    async fn call(self, txn: &Txn, key: Value, value: State) -> TCResult<()> {
        let (key_name, value_name, op_def) = self.method;

        let mut context = Map::new();
        context.insert(key_name, key.into());
        context.insert(value_name, value);

        call_method(txn, self.subject, self.path, context, op_def).await?;
        Ok(())
    }
}

impl<'a, T: Instance + Route + fmt::Display + 'a> Handler<'a> for PutMethod<'a, T>
where
    InstanceExt<T>: ToState,
{
    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(move |txn, key, value| {
            Box::pin(self.call(txn, key, value))
        }))
    }
}

struct PostMethod<'a, T: Instance> {
    subject: &'a InstanceExt<T>,
    method: PostOp,
    path: &'a [PathSegment],
}

impl<'a, T: Instance + Route + fmt::Display + 'a> PostMethod<'a, T>
where
    InstanceExt<T>: ToState,
{
    async fn call(self, txn: &Txn, params: Map<State>) -> TCResult<State> {
        call_method(txn, self.subject, self.path, params, self.method).await
    }
}

impl<'a, T: Instance + Route + fmt::Display + 'a> Handler<'a> for PostMethod<'a, T>
where
    InstanceExt<T>: ToState,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(move |txn, params| {
            Box::pin(self.call(txn, params))
        }))
    }
}

struct DeleteMethod<'a, T: Instance> {
    subject: &'a InstanceExt<T>,
    method: DeleteOp,
    path: &'a [PathSegment],
}

impl<'a, T: Instance + Route + fmt::Display + 'a> DeleteMethod<'a, T>
where
    InstanceExt<T>: ToState,
{
    async fn call(self, txn: &Txn, key: Value) -> TCResult<()> {
        let (key_name, op_def) = self.method;

        let mut context = Map::new();
        context.insert(key_name, key.into());

        call_method(txn, self.subject, self.path, context, op_def).await?;
        Ok(())
    }
}

impl<'a, T: Instance + Route + fmt::Display + 'a> Handler<'a> for DeleteMethod<'a, T>
where
    InstanceExt<T>: ToState,
{
    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(move |txn, key| Box::pin(self.call(txn, key))))
    }
}

impl<T: Instance + Route + fmt::Display> Route for InstanceExt<T>
where
    Self: ToState,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        debug!("{} route {} (parent is {} {})", self, TCPath::from(path), std::any::type_name::<T>(), self.parent());

        if path.is_empty() {
            debug!("routing to parent: {}", self.parent());
            self.parent().route(path)
        } else if let Some(attr) = self.proto().get(&path[0]) {
            debug!("{} found in instance proto", &path[0]);

            match attr {
                Scalar::Op(OpDef::Get(get_op)) => Some(Box::new(GetMethod {
                    subject: self,
                    method: get_op.clone(),
                    path: &path[1..],
                })),
                Scalar::Op(OpDef::Put(put_op)) => Some(Box::new(PutMethod {
                    subject: self,
                    method: put_op.clone(),
                    path: &path[1..],
                })),
                Scalar::Op(OpDef::Post(post_op)) => Some(Box::new(PostMethod {
                    subject: self,
                    method: post_op.clone(),
                    path: &path[1..],
                })),
                Scalar::Op(OpDef::Delete(delete_op)) => Some(Box::new(DeleteMethod {
                    subject: self,
                    method: delete_op.clone(),
                    path: &path[1..],
                })),
                other => other.route(&path[1..]),
            }
        } else if let Some(handler) = self.parent().route(path) {
            debug!("{} found in parent", TCPath::from(path));
            Some(handler)
        } else if let Some(attr) = self.proto().get(&path[0]) {
            debug!("{} found in class proto", path[0]);
            attr.route(&path[1..])
        } else {
            debug!("not found in {}: {}", self, TCPath::from(path));
            None
        }
    }
}

async fn call_method<T: Instance + Route + fmt::Display>(
    txn: &Txn,
    subject: &InstanceExt<T>,
    path: &[PathSegment],
    context: Map<State>,
    form: Vec<(Id, Scalar)>,
) -> TCResult<State>
where
    InstanceExt<T>: ToState,
{
    debug!(
        "call method with form {}",
        form.iter()
            .map(|(id, s)| format!("{}: {}", id, s))
            .collect::<Vec<String>>()
            .join("\n")
    );

    if !path.is_empty() {
        let msg = format!("{} member {}", subject, TCPath::from(path));
        return Err(TCError::not_found(msg));
    }

    let capture = if let Some((capture, _)) = form.last() {
        capture.clone()
    } else {
        return Ok(State::default());
    };

    Executor::with_context(txn, Some(subject), context.into(), form)
        .capture(capture)
        .await
}
