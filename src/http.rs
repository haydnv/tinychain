use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response};

const CONTENT_TYPE: &str = "application/json";

pub struct HTTPServer {
    addr: SocketAddr,
}

impl HTTPServer {
    pub fn new(addr: SocketAddr) -> Self {
        Self { addr }
    }

    async fn handle(_: Request<Body>) -> Result<Response<Body>, hyper::Error> {
        let response =
            destream_json::encode(scalar::Value::String("Hello, world!".into())).unwrap();

        let mut response = Response::new(Body::wrap_stream(response));
        response
            .headers_mut()
            .insert(hyper::header::CONTENT_TYPE, CONTENT_TYPE.parse().unwrap());

        Ok(response)
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
                Ok::<_, hyper::Error>(service_fn(HTTPServer::handle))
            }))
            .await
    }
}
