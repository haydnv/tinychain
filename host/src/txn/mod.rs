use std::sync::Arc;

use async_trait::async_trait;

use auth::Token;
use error::*;
use generic::Id;
use transact::fs::Dir;
use transact::{Transaction, TxnId};

use crate::fs;
use crate::gateway::Gateway;
use crate::scalar::{Link, Value};
use crate::state::State;

mod server;

pub use server::*;

pub struct Request {
    pub auth: Token,
    pub txn_id: TxnId,
}

impl Request {
    pub fn new(auth: Token, txn_id: TxnId) -> Self {
        Self { auth, txn_id }
    }

    pub fn contains(&self, other: &Self) -> bool {
        if self.txn_id == other.txn_id {
            self.auth.contains(&other.auth)
        } else {
            false
        }
    }
}

#[derive(Clone)]
struct Inner {
    dir: fs::Dir,
    gateway: Arc<Gateway>,
    request: Arc<Request>,
}

#[derive(Clone)]
pub struct Txn {
    inner: Arc<Inner>,
}

impl Txn {
    fn new(gateway: Arc<Gateway>, dir: fs::Dir, request: Request) -> Self {
        let request = Arc::new(request);
        let inner = Arc::new(Inner {
            dir,
            gateway,
            request,
        });

        Self { inner }
    }

    pub fn request(&'_ self) -> &'_ Request {
        &self.inner.request
    }

    pub async fn get(&self, link: Link, key: Value) -> TCResult<State> {
        self.inner.gateway.get(self, link, key).await
    }
}

#[async_trait]
impl Transaction<fs::Dir> for Txn {
    fn id(&'_ self) -> &'_ TxnId {
        &self.inner.request.txn_id
    }

    fn context(&'_ self) -> &'_ fs::Dir {
        &self.inner.dir
    }

    async fn subcontext(&self, id: Id) -> TCResult<Self> {
        let inner = Inner {
            gateway: self.inner.gateway.clone(),
            request: self.inner.request.clone(),
            dir: self.inner.dir.create_dir(id).await?,
        };

        Ok(Txn {
            inner: Arc::new(inner),
        })
    }
}
