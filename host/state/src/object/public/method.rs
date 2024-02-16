use std::fmt;
use std::marker::PhantomData;

use log::debug;

use tc_error::*;
use tc_scalar::op::*;
use tc_scalar::Scalar;
use tc_transact::public::{
    DeleteHandler, GetHandler, Handler, PostHandler, PutHandler, Route, ToState,
};
use tc_transact::{Gateway, Transaction};
use tc_value::Value;
use tcgeneric::{Id, Instance, Map, PathSegment};

use crate::{CacheBlock, State};

struct GetMethod<'a, Txn, T: Instance> {
    subject: &'a T,
    name: &'a Id,
    method: GetOp,
    phantom: PhantomData<Txn>,
}

impl<'a, Txn, T: Instance> GetMethod<'a, Txn, T> {
    pub fn new(subject: &'a T, name: &'a Id, method: GetOp) -> Self {
        Self {
            subject,
            name,
            method,
            phantom: PhantomData,
        }
    }
}

impl<'a, Txn, T> GetMethod<'a, Txn, T>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
    T: ToState<State<Txn>> + Instance + Route<State<Txn>> + fmt::Debug + 'a,
{
    async fn call(self, txn: &Txn, key: Value) -> TCResult<State<Txn>> {
        let (key_name, op_def) = self.method;

        let mut context = Map::new();
        context.insert(key_name, key.into());

        match call_method(txn, self.subject, context, op_def).await {
            Ok(state) => Ok(state),
            Err(cause) => {
                Err(cause.consume(format!("in call to GET {:?} /{}", self.subject, self.name)))
            }
        }
    }
}

impl<'a, Txn, T> Handler<'a, State<Txn>> for GetMethod<'a, Txn, T>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
    T: ToState<State<Txn>> + Instance + Route<State<Txn>> + fmt::Debug + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State<Txn>>>
    where
        'b: 'a,
    {
        Some(Box::new(move |txn, key| Box::pin(self.call(txn, key))))
    }
}

struct PutMethod<'a, Txn, T: Instance> {
    subject: &'a T,
    name: &'a Id,
    method: PutOp,
    phantom: PhantomData<Txn>,
}

impl<'a, Txn, T: Instance> PutMethod<'a, Txn, T> {
    pub fn new(subject: &'a T, name: &'a Id, method: PutOp) -> Self {
        Self {
            subject,
            name,
            method,
            phantom: PhantomData,
        }
    }
}

impl<'a, Txn, T> PutMethod<'a, Txn, T>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
    T: ToState<State<Txn>> + Instance + Route<State<Txn>> + fmt::Debug + 'a,
{
    async fn call(self, txn: &Txn, key: Value, value: State<Txn>) -> TCResult<()> {
        let (key_name, value_name, op_def) = self.method;

        let mut context = Map::new();
        context.insert(key_name, key.into());
        context.insert(value_name, value);

        match call_method(txn, self.subject, context, op_def).await {
            Ok(_) => Ok(()),
            Err(cause) => {
                Err(cause.consume(format!("in call to PUT {:?} /{}", self.subject, self.name)))
            }
        }
    }
}

impl<'a, Txn, T> Handler<'a, State<Txn>> for PutMethod<'a, Txn, T>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
    T: ToState<State<Txn>> + Instance + Route<State<Txn>> + fmt::Debug + 'a,
{
    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, Txn, State<Txn>>>
    where
        'b: 'a,
    {
        Some(Box::new(move |txn, key, value| {
            Box::pin(self.call(txn, key, value))
        }))
    }
}

struct PostMethod<'a, Txn, T: Instance> {
    subject: &'a T,
    name: &'a Id,
    method: PostOp,
    phantom: PhantomData<Txn>,
}

impl<'a, Txn, T: Instance> PostMethod<'a, Txn, T> {
    pub fn new(subject: &'a T, name: &'a Id, method: PostOp) -> Self {
        Self {
            subject,
            name,
            method,
            phantom: PhantomData,
        }
    }
}

impl<'a, Txn, T> PostMethod<'a, Txn, T>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
    T: ToState<State<Txn>> + Instance + Route<State<Txn>> + fmt::Debug + 'a,
{
    async fn call(self, txn: &Txn, params: Map<State<Txn>>) -> TCResult<State<Txn>> {
        match call_method(txn, self.subject, params, self.method).await {
            Ok(state) => Ok(state),
            Err(cause) => {
                Err(cause.consume(format!("in call to POST {:?} /{}", self.subject, self.name)))
            }
        }
    }
}

impl<'a, Txn, T> Handler<'a, State<Txn>> for PostMethod<'a, Txn, T>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
    T: ToState<State<Txn>> + Instance + Route<State<Txn>> + fmt::Debug + 'a,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State<Txn>>>
    where
        'b: 'a,
    {
        Some(Box::new(move |txn, params| {
            Box::pin(self.call(txn, params))
        }))
    }
}

struct DeleteMethod<'a, Txn, T: Instance> {
    subject: &'a T,
    name: &'a Id,
    method: DeleteOp,
    phantom: PhantomData<Txn>,
}

impl<'a, Txn, T: Instance> DeleteMethod<'a, Txn, T> {
    pub fn new(subject: &'a T, name: &'a Id, method: DeleteOp) -> Self {
        Self {
            subject,
            name,
            method,
            phantom: PhantomData,
        }
    }
}

impl<'a, Txn, T> DeleteMethod<'a, Txn, T>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
    T: ToState<State<Txn>> + Instance + Route<State<Txn>> + fmt::Debug + 'a,
{
    async fn call(self, txn: &Txn, key: Value) -> TCResult<()> {
        let (key_name, op_def) = self.method;

        let mut context = Map::new();
        context.insert(key_name, key.into());

        match call_method(txn, self.subject, context, op_def).await {
            Ok(_) => Ok(()),
            Err(cause) => Err(cause.consume(format!(
                "in call to DELETE {:?} /{}",
                self.subject, self.name
            ))),
        }
    }
}

impl<'a, Txn, T> Handler<'a, State<Txn>> for DeleteMethod<'a, Txn, T>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
    T: ToState<State<Txn>> + Instance + Route<State<Txn>> + fmt::Debug + 'a,
{
    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b, Txn>>
    where
        'b: 'a,
    {
        Some(Box::new(move |txn, key| Box::pin(self.call(txn, key))))
    }
}

#[inline]
pub fn route_attr<'a, Txn, T>(
    subject: &'a T,
    name: &'a Id,
    attr: &'a Scalar,
    path: &'a [PathSegment],
) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
    T: ToState<State<Txn>> + Instance + Route<State<Txn>> + fmt::Debug + 'a,
{
    match attr {
        Scalar::Op(OpDef::Get(get_op)) if path.is_empty() => {
            debug!("call GET method with subject {:?}", subject);

            Some(Box::new(GetMethod::new(subject, name, get_op.clone())))
        }
        Scalar::Op(OpDef::Put(put_op)) if path.is_empty() => {
            debug!("call PUT method with subject {:?}", subject);

            Some(Box::new(PutMethod::new(subject, name, put_op.clone())))
        }
        Scalar::Op(OpDef::Post(post_op)) if path.is_empty() => {
            debug!("call POST method with subject {:?}", subject);

            Some(Box::new(PostMethod::new(subject, name, post_op.clone())))
        }
        Scalar::Op(OpDef::Delete(delete_op)) if path.is_empty() => {
            debug!("call DELETE method with subject {:?}", subject);

            Some(Box::new(DeleteMethod::new(
                subject,
                name,
                delete_op.clone(),
            )))
        }
        other => other.route(path),
    }
}

async fn call_method<Txn, T>(
    txn: &Txn,
    subject: &T,
    context: Map<State<Txn>>,
    form: Vec<(Id, Scalar)>,
) -> TCResult<State<Txn>>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
    T: ToState<State<Txn>> + Route<State<Txn>> + Instance + fmt::Debug,
{
    debug!(
        "call method with form {}",
        form.iter()
            .map(|(id, s)| format!("{}: {:?}", id, s))
            .collect::<Vec<String>>()
            .join("\n")
    );

    let capture = if let Some((capture, _)) = form.last() {
        capture.clone()
    } else {
        return Ok(State::default());
    };

    Executor::with_context(txn, Some(subject), context.into(), form)
        .capture(capture)
        .await
}
