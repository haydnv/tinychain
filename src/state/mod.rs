use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::auth::Token;
use crate::error;
use crate::internal::file::File;
use crate::transaction::{Transact, Txn};
use crate::value::link::PathSegment;
use crate::value::{Args, TCResult, TCValue};

mod directory;
mod graph;
mod index;
pub mod table;
mod tensor;

pub type Directory = directory::Directory;
pub type Graph = graph::Graph;
pub type Tensor = tensor::Tensor;

#[async_trait]
pub trait Collection {
    type Key: TryFrom<TCValue>;
    type Value: TryFrom<TCValue>;

    async fn get(
        self: &Arc<Self>,
        txn: &Arc<Txn<'_>>,
        key: &Self::Key,
        auth: &Option<Token>,
    ) -> TCResult<Self::Value>;

    async fn put(
        self: Arc<Self>,
        txn: &Arc<Txn<'_>>,
        key: Self::Key,
        state: Self::Value,
        auth: &Option<Token>,
    ) -> TCResult<State>;
}

#[async_trait]
pub trait Derived: Collection {
    type Config: TryFrom<TCValue>;

    async fn create(txn: &Arc<Txn<'_>>, config: Self::Config) -> TCResult<Arc<Self>>;
}

#[async_trait]
pub trait Persistent: Collection + File {
    type Config: TryFrom<TCValue>;

    async fn create(txn: &Arc<Txn<'_>>, config: Self::Config) -> TCResult<Arc<Self>>;
}

#[derive(Clone)]
pub enum State {
    Directory(Arc<Directory>),
    Graph(Arc<Graph>),
    Table(Arc<table::Table>),
    Tensor(Arc<tensor::Tensor>),
    Value(TCValue),
}

impl State {
    pub async fn get(
        &self,
        txn: &Arc<Txn<'_>>,
        key: TCValue,
        auth: &Option<Token>,
    ) -> TCResult<State> {
        match self {
            State::Directory(d) => d.clone().get(txn, &key.try_into()?, auth).await,
            State::Graph(g) => Ok(g.clone().get(txn, &key, auth).await?.into()),
            State::Table(t) => Ok(t.clone().get(txn, &key.try_into()?, auth).await?.into()),
            _ => Err(error::bad_request(
                &format!("Cannot GET {} from", key),
                self,
            )),
        }
    }

    pub fn is_value(&self) -> bool {
        match self {
            State::Value(_) => true,
            _ => false,
        }
    }

    pub async fn put(
        &self,
        txn: &Arc<Txn<'_>>,
        key: TCValue,
        value: TCValue,
        auth: &Option<Token>,
    ) -> TCResult<State> {
        match self {
            State::Directory(d) => {
                d.clone()
                    .put(txn, key.try_into()?, value.try_into()?, auth)
                    .await
            }
            State::Graph(g) => g.clone().put(txn, key, value, auth).await,
            State::Table(t) => {
                t.clone()
                    .put(txn, key.try_into()?, value.try_into()?, auth)
                    .await
            }
            _ => Err(error::bad_request("Cannot PUT to", self)),
        }
    }

    pub async fn post(
        &self,
        _txn: Arc<Txn<'_>>,
        _method: &PathSegment,
        _args: Args,
        _auth: &Option<Token>,
    ) -> TCResult<State> {
        Err(error::method_not_allowed(format!(
            "{} does not support POST",
            self
        )))
    }
}

impl From<Arc<Directory>> for State {
    fn from(dir: Arc<Directory>) -> State {
        State::Directory(dir)
    }
}

impl From<Arc<Graph>> for State {
    fn from(graph: Arc<Graph>) -> State {
        State::Graph(graph)
    }
}

impl From<Arc<table::Table>> for State {
    fn from(table: Arc<table::Table>) -> State {
        State::Table(table)
    }
}

impl<T: Into<TCValue>> From<T> for State {
    fn from(value: T) -> State {
        State::Value(value.into())
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            State::Directory(_) => write!(f, "(directory)"),
            State::Graph(_) => write!(f, "(graph)"),
            State::Table(_) => write!(f, "(table)"),
            State::Tensor(_) => write!(f, "(tensor)"),
            State::Value(value) => write!(f, "value: {}", value),
        }
    }
}
