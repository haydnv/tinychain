use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::mpsc;

use error::*;
use generic::Id;
use transact::fs::Dir;
pub use transact::{Transaction, TxnId};

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
    gateway: Arc<Gateway>,
    request: Arc<Request>,
    txn_server: mpsc::UnboundedSender<TxnId>,
}

#[derive(Clone)]
pub struct Txn {
    inner: Arc<Inner>,
    dir: fs::Dir,
}

impl Txn {
    fn new(
        txn_server: mpsc::UnboundedSender<TxnId>,
        gateway: Arc<Gateway>,
        dir: fs::Dir,
        request: Request,
    ) -> Self {
        let request = Arc::new(request);
        let inner = Arc::new(Inner {
            gateway,
            request,
            txn_server,
        });

        Self { inner, dir }
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
        &self.dir
    }

    async fn subcontext(&self, id: Id) -> TCResult<Self> {
        let dir = self.dir.create_dir(self.inner.request.txn_id, id).await?;

        Ok(Txn {
            inner: self.inner.clone(),
            dir,
        })
    }
}

impl Drop for Txn {
    fn drop(&mut self) {
        // There will still be one reference in TxnServer when all others are dropped, plus this one
        if Arc::strong_count(&self.inner) == 2 {
            self.inner
                .txn_server
                .send(self.inner.request.txn_id)
                .unwrap();
        }
    }
}
