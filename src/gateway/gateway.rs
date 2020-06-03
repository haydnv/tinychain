use std::collections::HashSet;

use futures::Stream;

use crate::auth::Token;
use crate::error;
use crate::kernel::Kernel;
use crate::state::State;
use crate::transaction::TxnId;
use crate::value::link::Link;
use crate::value::{TCResult, Value, ValueId};

use super::op;
use super::Hosted;
use super::NetworkTime;

pub struct Gateway {
    kernel: Kernel,
    hosted: Hosted,
}

impl Gateway {
    pub fn new(kernel: Kernel, hosted: Hosted) -> Gateway {
        Gateway { kernel, hosted }
    }

    pub async fn authenticate(&self, _token: &str) -> TCResult<Token> {
        Err(error::not_implemented())
    }

    pub async fn resolve(&self, _link: Link) -> TCResult<State> {
        Err(error::not_implemented())
    }

    pub fn txn_id(&self) -> TxnId {
        TxnId::new(NetworkTime::now())
    }

    // /transact/execute
    pub async fn execute<I: Stream<Item = (ValueId, Value)>>(
        &self,
        _auth: Option<Token>,
        _capture: HashSet<ValueId>,
        _request: op::Post<I>,
    ) -> TCResult<State> {
        Err(error::not_implemented())
    }

    // TODO: /transact/hypothetical, /transact/explain, /transact/background

    pub async fn get(&self, _subject: op::Subject, _op: op::Get) -> TCResult<State> {
        Err(error::not_implemented())
    }

    pub async fn put<S: Stream<Item = (Value, Value)>>(
        &self,
        _subject: op::Subject,
        _op: op::Put<S>,
    ) -> TCResult<State> {
        Err(error::not_implemented())
    }
}
