use std::convert::TryFrom;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error;
use crate::state::State;
use crate::transaction::Txn;
use crate::value::{Args, PathSegment, TCResult, TCValue};

mod actor;

pub type Actor = actor::Actor;

#[async_trait]
pub trait TCObject: Into<TCValue> + TryFrom<TCValue> {
    fn class() -> &'static str;

    fn id(&self) -> TCValue;

    async fn post(
        &self,
        _txn: Arc<Txn>,
        _method: &PathSegment,
        mut _args: Args,
    ) -> TCResult<State> {
        Err(error::method_not_allowed(self.id()))
    }
}

#[derive(Clone)]
pub enum Object {
    Actor(Arc<Actor>),
}

impl Object {
    pub fn class(&self) -> &'static str {
        match self {
            Object::Actor(_) => Actor::class(),
        }
    }

    pub async fn post(&self, txn: Arc<Txn>, method: &PathSegment, args: Args) -> TCResult<State> {
        match self {
            Object::Actor(actor) => actor.post(txn, method, args).await,
        }
    }
}

impl From<Arc<Actor>> for Object {
    fn from(a: Arc<Actor>) -> Object {
        Object::Actor(a)
    }
}
