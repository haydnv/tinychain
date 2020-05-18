use std::convert::TryFrom;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error;
use crate::object::actor::Token;
use crate::state::State;
use crate::transaction::Txn;
use crate::value::link::PathSegment;
use crate::value::{Args, TCResult, TCValue};

pub mod actor;

#[async_trait]
pub trait TCObject: Into<TCValue> + TryFrom<TCValue> {
    fn class() -> &'static str;

    fn id(&self) -> TCValue;

    async fn post(
        &self,
        _txn: Arc<Txn<'_>>,
        _method: &PathSegment,
        mut _args: Args,
        _auth: &Option<Token>,
    ) -> TCResult<State> {
        Err(error::method_not_allowed(self.id()))
    }
}

#[derive(Clone)]
pub enum Object {
    Actor(Arc<actor::Actor>),
}

impl Object {
    pub fn class(&self) -> &'static str {
        match self {
            Object::Actor(_) => actor::Actor::class(),
        }
    }

    pub async fn post(
        &self,
        txn: Arc<Txn<'_>>,
        method: &PathSegment,
        args: Args,
        auth: &Option<Token>,
    ) -> TCResult<State> {
        match self {
            Object::Actor(actor) => actor.post(txn, method, args, auth).await,
        }
    }
}

impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Object::Actor(actor) => write!(f, "{}", actor),
        }
    }
}

impl From<Arc<actor::Actor>> for Object {
    fn from(a: Arc<actor::Actor>) -> Object {
        Object::Actor(a)
    }
}

impl TryFrom<Object> for Arc<actor::Actor> {
    type Error = error::TCError;

    fn try_from(object: Object) -> TCResult<Arc<actor::Actor>> {
        match object {
            Object::Actor(actor) => Ok(actor),
        }
    }
}
