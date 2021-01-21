use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response};

const HELLO: &[u8] = b"Hello, world!";

#[derive(Clone)]
pub struct HTTPServer {
    addr: SocketAddr,
}

impl HTTPServer {
    pub fn new(addr: SocketAddr) -> Self {
        Self { addr }
    }

    async fn route(_: Request<Body>) -> Result<Response<Body>, hyper::Error> {
        Ok(Response::new(Body::from(HELLO)))
    }
}

#[async_trait]
impl super::Server for HTTPServer {
    type Error = hyper::Error;

    async fn listen(self) -> Result<(), Self::Error> {
        println!("HTTP server listening on {}", self.addr);
        let this = Arc::new(self);

        hyper::Server::bind(&this.addr)
            .serve(make_service_fn(|_| async {
                Ok::<_, hyper::Error>(service_fn(HTTPServer::route))
            }))
            .await
    }
}
