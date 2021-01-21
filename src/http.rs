use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, StatusCode};

use error::TCError;

const CONTENT_TYPE: &str = "application/json";

pub struct HTTPServer {
    addr: SocketAddr,
}

impl HTTPServer {
    pub fn new(addr: SocketAddr) -> Self {
        Self { addr }
    }

    async fn handle(_: Request<Body>) -> Result<Response<Body>, hyper::Error> {
        let response = destream_json::encode(scalar::Value::String("Hello, world!".into()))
            .map_err(TCError::internal);

        let response = match response {
            Ok(response_data) => {
                let mut response = Response::new(Body::wrap_stream(response_data));
                response
                    .headers_mut()
                    .insert(hyper::header::CONTENT_TYPE, CONTENT_TYPE.parse().unwrap());

                response
            }
            Err(cause) => transform_error(cause),
        };

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

fn transform_error(err: error::TCError) -> hyper::Response<Body> {
    let mut response = hyper::Response::new(Body::from(format!("{}\r\n", err.message())));

    use error::ErrorType::*;
    *response.status_mut() = match err.code() {
        BadRequest => StatusCode::BAD_REQUEST,
        Forbidden => StatusCode::FORBIDDEN,
        Internal => StatusCode::INTERNAL_SERVER_ERROR,
        MethodNotAllowed => StatusCode::METHOD_NOT_ALLOWED,
        Timeout => StatusCode::REQUEST_TIMEOUT,
        Unauthorized => StatusCode::UNAUTHORIZED,
    };

    response
}
