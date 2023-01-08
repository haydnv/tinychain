use std::fmt;

use log::debug;

use tc_error::*;
use tc_value::Value;
use tcgeneric::{Id, Instance, Map, PathSegment};

use crate::route::{DeleteHandler, GetHandler, Handler, PostHandler, PutHandler, Route};
use crate::scalar::op::*;
use crate::scalar::Scalar;
use crate::state::{State, ToState};
use crate::txn::Txn;

struct GetMethod<'a, T: Instance> {
    subject: &'a T,
    name: &'a Id,
    method: GetOp,
}

impl<'a, T: Instance> GetMethod<'a, T> {
    pub fn new(subject: &'a T, name: &'a Id, method: GetOp) -> Self {
        Self {
            subject,
            name,
            method,
        }
    }
}

impl<'a, T: ToState + Instance + Route + fmt::Display + 'a> GetMethod<'a, T> {
    async fn call(self, txn: &Txn, key: Value) -> TCResult<State> {
        let (key_name, op_def) = self.method;

        let mut context = Map::new();
        context.insert(key_name, key.into());

        match call_method(txn, self.subject, context, op_def).await {
            Ok(state) => Ok(state),
            Err(cause) => {
                Err(cause.consume(format!("in call to GET {} /{}", self.subject, self.name)))
            }
        }
    }
}

impl<'a, T: ToState + Instance + Route + fmt::Display + 'a> Handler<'a> for GetMethod<'a, T> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(move |txn, key| Box::pin(self.call(txn, key))))
    }
}

struct PutMethod<'a, T: Instance> {
    subject: &'a T,
    name: &'a Id,
    method: PutOp,
}

impl<'a, T: Instance> PutMethod<'a, T> {
    pub fn new(subject: &'a T, name: &'a Id, method: PutOp) -> Self {
        Self {
            subject,
            name,
            method,
        }
    }
}

impl<'a, T: ToState + Instance + Route + fmt::Display + 'a> PutMethod<'a, T> {
    async fn call(self, txn: &Txn, key: Value, value: State) -> TCResult<()> {
        let (key_name, value_name, op_def) = self.method;

        let mut context = Map::new();
        context.insert(key_name, key.into());
        context.insert(value_name, value);

        match call_method(txn, self.subject, context, op_def).await {
            Ok(_) => Ok(()),
            Err(cause) => {
                Err(cause.consume(format!("in call to PUT {} /{}", self.subject, self.name)))
            }
        }
    }
}

impl<'a, T: ToState + Instance + Route + fmt::Display + 'a> Handler<'a> for PutMethod<'a, T> {
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
    subject: &'a T,
    name: &'a Id,
    method: PostOp,
}

impl<'a, T: Instance> PostMethod<'a, T> {
    pub fn new(subject: &'a T, name: &'a Id, method: PostOp) -> Self {
        Self {
            subject,
            name,
            method,
        }
    }
}

impl<'a, T: ToState + Instance + Route + fmt::Display + 'a> PostMethod<'a, T> {
    async fn call(self, txn: &Txn, params: Map<State>) -> TCResult<State> {
        match call_method(txn, self.subject, params, self.method).await {
            Ok(state) => Ok(state),
            Err(cause) => {
                Err(cause.consume(format!("in call to POST {} /{}", self.subject, self.name)))
            }
        }
    }
}

impl<'a, T: ToState + Instance + Route + fmt::Display + 'a> Handler<'a> for PostMethod<'a, T> {
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
    subject: &'a T,
    name: &'a Id,
    method: DeleteOp,
}

impl<'a, T: Instance> DeleteMethod<'a, T> {
    pub fn new(subject: &'a T, name: &'a Id, method: DeleteOp) -> Self {
        Self {
            subject,
            name,
            method,
        }
    }
}

impl<'a, T: ToState + Instance + Route + fmt::Display + 'a> DeleteMethod<'a, T> {
    async fn call(self, txn: &Txn, key: Value) -> TCResult<()> {
        let (key_name, op_def) = self.method;

        let mut context = Map::new();
        context.insert(key_name, key.into());

        match call_method(txn, self.subject, context, op_def).await {
            Ok(_) => Ok(()),
            Err(cause) => {
                Err(cause.consume(format!("in call to DELETE {} /{}", self.subject, self.name)))
            }
        }
    }
}

impl<'a, T: ToState + Instance + Route + fmt::Display + 'a> Handler<'a> for DeleteMethod<'a, T> {
    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(move |txn, key| Box::pin(self.call(txn, key))))
    }
}

#[inline]
pub fn route_attr<'a, T>(
    subject: &'a T,
    name: &'a Id,
    attr: &'a Scalar,
    path: &'a [PathSegment],
) -> Option<Box<dyn Handler<'a> + 'a>>
where
    T: ToState + Instance + Route + fmt::Display + 'a,
{
    match attr {
        Scalar::Op(OpDef::Get(get_op)) if path.is_empty() => {
            debug!("call GET method with subject {}", subject);

            Some(Box::new(GetMethod::new(subject, name, get_op.clone())))
        }
        Scalar::Op(OpDef::Put(put_op)) if path.is_empty() => {
            debug!("call PUT method with subject {}", subject);

            Some(Box::new(PutMethod::new(subject, name, put_op.clone())))
        }
        Scalar::Op(OpDef::Post(post_op)) if path.is_empty() => {
            debug!("call POST method with subject {}", subject);

            Some(Box::new(PostMethod::new(subject, name, post_op.clone())))
        }
        Scalar::Op(OpDef::Delete(delete_op)) if path.is_empty() => {
            debug!("call DELETE method with subject {}", subject);

            Some(Box::new(DeleteMethod::new(
                subject,
                name,
                delete_op.clone(),
            )))
        }
        other => other.route(path),
    }
}

async fn call_method<T: ToState + Route + Instance + fmt::Display>(
    txn: &Txn,
    subject: &T,
    context: Map<State>,
    form: Vec<(Id, Scalar)>,
) -> TCResult<State> {
    debug!(
        "call method with form {}",
        form.iter()
            .map(|(id, s)| format!("{}: {}", id, s))
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
