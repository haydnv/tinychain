use std::sync::Arc;

use futures::{future, stream, Stream};

use crate::auth::{Auth, Token};
use crate::error;
use crate::kernel;
use crate::state::{Dir, GetResult, State};
use crate::transaction::{Txn, TxnContext, TxnId};
use crate::value::link::Link;
use crate::value::{TCResult, Value, ValueId};

use super::{Hosted, NetworkTime};

pub struct Gateway {
    hosted: Hosted,
    workspace: Arc<Dir>,
}

impl Gateway {
    pub fn time() -> NetworkTime {
        NetworkTime::now()
    }

    pub fn new(hosted: Hosted, workspace: Arc<Dir>) -> Arc<Gateway> {
        Arc::new(Gateway { hosted, workspace })
    }

    pub async fn authenticate(&self, _token: &str) -> TCResult<Token> {
        Err(error::not_implemented())
    }

    pub async fn transaction(self: &Arc<Self>) -> TCResult<Arc<Txn>> {
        Txn::new(self.clone(), self.workspace.clone()).await
    }

    pub async fn get(
        &self,
        subject: &Link,
        selector: Value,
        _auth: &Auth,
        _txn_id: Option<TxnId>,
    ) -> GetResult {
        if subject.host().is_none() {
            let path = subject.path();
            if path[0] == "sbin" {
                let state = kernel::get(&path.slice_from(1), selector)?;
                Ok(Box::pin(stream::once(future::ready(state))))
            } else {
                Err(error::not_implemented())
            }
        } else if let Some((_rel_path, _cluster)) = self.hosted.get(subject.path()) {
            Err(error::not_implemented())
        } else {
            Err(error::not_implemented())
        }
    }

    pub async fn put(
        &self,
        _subject: &Link,
        _selector: Value,
        _state: State,
        _auth: &Auth,
    ) -> TCResult<State> {
        Err(error::not_implemented())
    }

    pub async fn post<S: Stream<Item = (ValueId, Value)>>(
        &self,
        subject: &Link,
        op: S,
        auth: &Auth,
    ) -> TCResult<TxnContext> {
        if subject.host().is_none() {
            kernel::post(subject.path(), op, auth).await
        } else {
            Err(error::not_implemented())
        }
    }
}
