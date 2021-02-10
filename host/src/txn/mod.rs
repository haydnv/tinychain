use std::sync::Arc;

use async_trait::async_trait;

use error::*;
use generic::Id;
use transact::fs::Dir;
use transact::{Transaction, TxnId};

use crate::fs;
use crate::gateway::Gateway;
use crate::scalar::{Link, Value};
use crate::state::State;

mod request;
mod server;

pub use request::*;
pub use server::*;

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

    pub async fn put(&self, link: Link, key: Value, value: State) -> TCResult<()> {
        self.inner.gateway.put(self, link, key, value).await
    }

    pub async fn post(&self, link: Link, params: State) -> TCResult<State> {
        self.inner.gateway.post(self, link, params).await
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
            dir: self
                .inner
                .dir
                .create_dir(self.inner.request.txn_id, id)
                .await?,
        };

        Ok(Txn {
            inner: Arc::new(inner),
        })
    }
}
