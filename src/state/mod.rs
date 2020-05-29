use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::auth::Token;
use crate::error;
use crate::internal::block::Store;
use crate::internal::file::File;
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::link::PathSegment;
use crate::value::op::PutOp;
use crate::value::{TCResult, TCValue, ValueId};

mod directory;
mod graph;
mod index;
pub mod table;
mod tensor;

pub type Directory = directory::Directory;
pub type Graph = graph::Graph;

#[async_trait]
pub trait Authorized: Collection {
    async fn get(
        self: &Arc<Self>,
        txn: &Arc<Txn<'_>>,
        key: &Self::Key,
        _auth: &Option<Token>,
    ) -> TCResult<Self::Value> {
        // TODO: authorize
        Collection::get(self, txn, key).await
    }

    async fn put(
        self: Arc<Self>,
        txn: &Arc<Txn<'_>>,
        key: Self::Key,
        state: Self::Value,
        _auth: &Option<Token>,
    ) -> TCResult<State> {
        // TODO: authorize
        Collection::put(self, txn, key, state).await
    }
}

#[async_trait]
pub trait Collection: Send + Sync {
    type Key: TryFrom<TCValue> + Send + Sync;
    type Value: TryFrom<TCValue> + Send + Sync;

    async fn get(self: &Arc<Self>, txn: &Arc<Txn<'_>>, key: &Self::Key) -> TCResult<Self::Value>;

    async fn put(
        self: Arc<Self>,
        txn: &Arc<Txn<'_>>,
        key: Self::Key,
        state: Self::Value,
    ) -> TCResult<State>;
}

#[async_trait]
pub trait Derived: Collection + Extend<PutOp> + Sized {
    type Config: TryFrom<Args>;

    async fn new(txn_id: &TxnId, context: Arc<Store>, config: Self::Config) -> TCResult<Self>;
}

#[async_trait]
pub trait Persistent: Collection + File {
    type Config: TryFrom<Args>;

    async fn create(txn: &Arc<Txn<'_>>, config: Self::Config) -> TCResult<Arc<Self>>;
}

pub struct Args(HashMap<ValueId, TCValue>);

impl Args {
    fn assert_empty(&self) -> TCResult<()> {
        if self.0.is_empty() {
            Ok(())
        } else {
            let keys: Vec<String> = self.0.keys().map(|v| v.to_string()).collect();
            Err(error::bad_request(
                "Unrecognized arguments provided",
                keys[..].join(", "),
            ))
        }
    }

    fn take<E: Into<error::TCError>, T: TryFrom<TCValue, Error = E>>(
        &mut self,
        name: &str,
    ) -> TCResult<T> {
        if let Some(value) = self.0.remove(&name.parse()?) {
            value.try_into().map_err(|e: E| e.into())
        } else {
            Err(error::bad_request("Required argument not provided", name))
        }
    }

    fn take_or<E: Into<error::TCError>, T: TryFrom<TCValue, Error = E>>(
        &mut self,
        name: &str,
        default: T,
    ) -> TCResult<T> {
        if let Some(value) = self.0.remove(&name.parse()?) {
            value.try_into().map_err(|e: E| e.into())
        } else {
            Ok(default)
        }
    }
}

impl TryFrom<TCValue> for Args {
    type Error = error::TCError;

    fn try_from(value: TCValue) -> TCResult<Args> {
        let args: HashMap<ValueId, TCValue> = value.try_into()?;
        Ok(Args(args))
    }
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
        _auth: &Option<Token>,
    ) -> TCResult<State> {
        // TODO: authorize
        match self {
            State::Directory(d) => d.clone().get(txn, &key.try_into()?).await,
            State::Graph(g) => Ok(g.clone().get(txn, &key).await?.into()),
            State::Table(t) => Ok(t.clone().get(txn, &key.try_into()?).await?.into()),
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
        _auth: &Option<Token>,
    ) -> TCResult<State> {
        // TODO: authorize
        match self {
            State::Directory(d) => d.clone().put(txn, key.try_into()?, value.try_into()?).await,
            State::Graph(g) => g.clone().put(txn, key, value).await,
            State::Table(t) => t.clone().put(txn, key.try_into()?, value.try_into()?).await,
            _ => Err(error::bad_request("Cannot PUT to", self)),
        }
    }

    pub async fn post(
        &self,
        _txn: Arc<Txn<'_>>,
        _method: &PathSegment,
        _args: Vec<(ValueId, TCValue)>,
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
