use std::net::SocketAddr;
use std::sync::Arc;

use error::*;
use generic::NetworkTime;

use crate::kernel::Kernel;
use crate::scalar::{Link, Value};
use crate::state::State;
use crate::TxnId;

#[async_trait::async_trait]
pub trait Server {
    type Error: std::error::Error;

    async fn listen(self, addr: SocketAddr) -> Result<(), Self::Error>;
}

pub struct Gateway {
    kernel: Kernel,
}

impl Gateway {
    pub fn time() -> NetworkTime {
        NetworkTime::now()
    }

    pub fn new() -> Arc<Self> {
        Arc::new(Self { kernel: Kernel })
    }

    pub async fn get(&self, txn_id: TxnId, subject: Link, key: Value) -> TCResult<State> {
        if subject.host().is_none() {
            self.kernel.get(txn_id, subject.path(), key).await
        } else {
            Err(TCError::not_implemented("remote GET"))
        }
    }

    #[cfg(feature = "http")]
    pub fn http_listen(
        &self,
        address: std::net::IpAddr,
        port: u16,
    ) -> std::pin::Pin<Box<impl futures::Future<Output = Result<(), Box<dyn std::error::Error>>>>>
    {
        use futures::future::TryFutureExt;

        let http_addr = (address, port).into();
        let server = crate::http::HTTPServer::new(Kernel);
        let listener = server.listen(http_addr).map_err(|e| {
            let e: Box<dyn std::error::Error> = Box::new(e);
            e
        });

        Box::pin(listener)
    }
}
