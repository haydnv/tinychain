use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use futures::{TryFutureExt, TryStreamExt};
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response};
use log::debug;
use serde::de::DeserializeOwned;
use transact::TxnId;

use error::*;
use generic::{NetworkTime, TCPathBuf};

use crate::kernel::Kernel;
use crate::state::State;

const CONTENT_TYPE: &str = "application/json";

pub struct HTTPServer {
    kernel: Kernel,
}

impl HTTPServer {
    pub fn new(kernel: Kernel) -> Self {
        Self { kernel }
    }

    async fn handle(
        self: Arc<Self>,
        request: Request<Body>,
    ) -> Result<Response<Body>, hyper::Error> {
        match self.route(request).await {
            Ok(state) => match destream_json::encode(state) {
                Ok(response) => {
                    let mut response = Response::new(Body::wrap_stream(response));
                    response
                        .headers_mut()
                        .insert(hyper::header::CONTENT_TYPE, CONTENT_TYPE.parse().unwrap());

                    Ok(response)
                }
                Err(cause) => Ok(transform_error(TCError::internal(cause))),
            },
            Err(cause) => Ok(transform_error(cause)),
        }
    }

    async fn route(&self, request: hyper::Request<Body>) -> TCResult<State> {
        let path: TCPathBuf = request.uri().path().parse()?;

        let mut params = request
            .uri()
            .query()
            .map(|v| {
                debug!("param {}", v);
                url::form_urlencoded::parse(v.as_bytes())
                    .into_owned()
                    .collect()
            })
            .unwrap_or_else(HashMap::new);

        let txn_id = if let Some(txn_id) = get_param(&mut params, "txn_id")? {
            txn_id
        } else {
            TxnId::new(NetworkTime::now())
        };

        match request.method() {
            &hyper::Method::GET => {
                let key = get_param(&mut params, "key")?.unwrap_or_default();
                self.kernel.get(txn_id, &path, key).await
            }
            &hyper::Method::POST => {
                let data = request.into_body().map_ok(|bytes| bytes.to_vec());
                let data = destream_json::try_decode(data)
                    .map_err(|e| TCError::bad_request("error deserializing POST data", e))
                    .await?;

                self.kernel.post(txn_id, &path, data).await
            }
            other => Err(TCError::method_not_allowed(other)),
        }
    }
}

#[async_trait]
impl crate::gateway::Server for HTTPServer {
    type Error = hyper::Error;

    async fn listen(self, addr: SocketAddr) -> Result<(), Self::Error> {
        println!("HTTP server listening on {}", &addr);
        let server = Arc::new(self);

        let new_service = make_service_fn(move |_| {
            let server = server.clone();
            async {
                Ok::<_, hyper::Error>(service_fn(move |req| {
                    let server = server.clone();
                    HTTPServer::handle(server, req)
                }))
            }
        });

        hyper::Server::bind(&addr).serve(new_service).await
    }
}

fn get_param<T: DeserializeOwned>(
    params: &mut HashMap<String, String>,
    name: &str,
) -> TCResult<Option<T>> {
    if let Some(param) = params.remove(name) {
        let val: T = serde_json::from_str(&param).map_err(|e| {
            TCError::bad_request(&format!("Unable to parse URI parameter '{}'", name), e)
        })?;

        Ok(Some(val))
    } else {
        Ok(None)
    }
}

fn transform_error(err: TCError) -> hyper::Response<Body> {
    let mut response = hyper::Response::new(Body::from(format!("{}\r\n", err.message())));

    use error::ErrorType::*;
    use hyper::StatusCode;
    *response.status_mut() = match err.code() {
        BadRequest => StatusCode::BAD_REQUEST,
        Forbidden => StatusCode::FORBIDDEN,
        Conflict => StatusCode::CONFLICT,
        Internal => StatusCode::INTERNAL_SERVER_ERROR,
        MethodNotAllowed => StatusCode::METHOD_NOT_ALLOWED,
        NotFound => StatusCode::NOT_FOUND,
        NotImplemented => StatusCode::NOT_IMPLEMENTED,
        Timeout => StatusCode::REQUEST_TIMEOUT,
        Unauthorized => StatusCode::UNAUTHORIZED,
    };

    response
}
