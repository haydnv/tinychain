use std::sync::Arc;

use futures::{future, stream, Stream};

use crate::auth::{Auth, Token};
use crate::error;
use crate::internal::Dir;
use crate::kernel;
use crate::state::{GetResult, State};
use crate::transaction::{Txn, TxnContext};
use crate::value::link::Link;
use crate::value::{TCResult, Value, ValueId};

use super::{Hosted, NetworkTime};

pub struct Gateway {
    hosted: Hosted,
    workspace: Arc<Dir>,
}

impl Gateway {
    pub fn new(hosted: Hosted, workspace: Arc<Dir>) -> Arc<Gateway> {
        Arc::new(Gateway { hosted, workspace })
    }

    pub async fn authenticate(&self, _token: &str) -> TCResult<Token> {
        Err(error::not_implemented())
    }

    pub fn time(&self) -> NetworkTime {
        NetworkTime::now()
    }

    pub async fn transaction<'a>(self: &Arc<Self>) -> TCResult<Arc<Txn>> {
        Txn::new(self.clone(), self.workspace.clone()).await
    }

    pub async fn get(&self, subject: &Link, selector: Value, _auth: &Auth) -> GetResult {
        if subject.host().is_none() {
            let state = kernel::get(subject.path(), selector)?;
            Ok(Box::pin(stream::once(future::ready(state))))
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
