use std::fmt;

use futures::future;
use log::{debug, info};

use tc_error::*;
use tc_value::Value;
use tcgeneric::{Id, Instance, Map, PathSegment, TCPath};

use crate::object::InstanceExt;
use crate::route::{DeleteHandler, GetHandler, Handler, PostHandler, PutHandler, Route, COPY};
use crate::scalar::*;
use crate::state::{State, ToState};
use crate::txn::Txn;

struct CopyHandler<'a, T> {
    instance: &'a T,
}

impl<'a, T> Handler<'a> for CopyHandler<'a, T>
where
    T: Instance + fmt::Display + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(move |_txn, _key| {
            Box::pin(future::ready(Err(TCError::not_implemented(format!(
                "{} has no /copy method",
                self.instance
            )))))
        }))
    }
}

impl<'a, T> From<&'a T> for CopyHandler<'a, T> {
    fn from(instance: &'a T) -> Self {
        Self { instance }
    }
}

struct GetMethod<'a, T: Instance> {
    subject: &'a InstanceExt<T>,
    name: &'a Id,
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

        match call_method(txn, self.subject, self.path, context, op_def).await {
            Ok(state) => Ok(state),
            Err(cause) => {
                Err(cause.consume(format!("in call to GET {} /{}", self.subject, self.name)))
            }
        }
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
    name: &'a Id,
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

        match call_method(txn, self.subject, self.path, context, op_def).await {
            Ok(_) => Ok(()),
            Err(cause) => {
                Err(cause.consume(format!("in call to PUT {} /{}", self.subject, self.name)))
            }
        }
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
    name: &'a Id,
    method: PostOp,
    path: &'a [PathSegment],
}

impl<'a, T: Instance + Route + fmt::Display + 'a> PostMethod<'a, T>
where
    InstanceExt<T>: ToState,
{
    async fn call(self, txn: &Txn, params: Map<State>) -> TCResult<State> {
        match call_method(txn, self.subject, self.path, params, self.method).await {
            Ok(state) => Ok(state),
            Err(cause) => {
                Err(cause.consume(format!("in call to POST {} /{}", self.subject, self.name)))
            }
        }
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
    name: &'a Id,
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

        match call_method(txn, self.subject, self.path, context, op_def).await {
            Ok(_) => Ok(()),
            Err(cause) => {
                Err(cause.consume(format!("in call to DELETE {} /{}", self.subject, self.name)))
            }
        }
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
        debug!(
            "{} route {} (parent is {} {})",
            self,
            TCPath::from(path),
            std::any::type_name::<T>(),
            self.parent()
        );

        if path.is_empty() {
            debug!("routing to parent: {}", self.parent());

            if let Some(handler) = self.parent().route(path) {
                Some(handler)
            } else if path == &COPY[..] {
                info!("tried to copy an object with no /copy method implemented");
                Some(Box::new(CopyHandler::from(self)))
            } else {
                debug!(
                    "instance {} has no handler for {}",
                    self,
                    TCPath::from(path)
                );
                None
            }
        } else if let Some(attr) = self.members().get(&path[0]) {
            debug!("{} found in instance members", &path[0]);

            if let State::Scalar(attr) = attr {
                route_attr(self, &path[0], attr, &path[1..])
            } else {
                attr.route(&path[1..])
            }
        } else if let Some(attr) = self.proto().get(&path[0]) {
            debug!("{} found in instance proto", &path[0]);
            route_attr(self, &path[0], attr, &path[1..])
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

#[inline]
fn route_attr<'a, T>(
    subject: &'a InstanceExt<T>,
    name: &'a Id,
    attr: &'a Scalar,
    path: &'a [PathSegment],
) -> Option<Box<dyn Handler<'a> + 'a>>
where
    T: Instance + Route + fmt::Display + 'a,
    InstanceExt<T>: ToState,
{
    match attr {
        Scalar::Op(OpDef::Get(get_op)) => {
            debug!("call GET method");

            Some(Box::new(GetMethod {
                subject,
                name,
                method: get_op.clone(),
                path,
            }))
        }
        Scalar::Op(OpDef::Put(put_op)) => {
            debug!("call PUT method");

            Some(Box::new(PutMethod {
                subject,
                name,
                method: put_op.clone(),
                path,
            }))
        }
        Scalar::Op(OpDef::Post(post_op)) => {
            debug!("call POST method");

            Some(Box::new(PostMethod {
                subject,
                name,
                method: post_op.clone(),
                path,
            }))
        }
        Scalar::Op(OpDef::Delete(delete_op)) => {
            debug!("call DELETE method");

            Some(Box::new(DeleteMethod {
                subject,
                name,
                method: delete_op.clone(),
                path,
            }))
        }
        other => other.route(path),
    }
}
