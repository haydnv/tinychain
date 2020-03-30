use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;

use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Method, Request, Response, Server, StatusCode};
use serde::{Deserialize, Serialize};

use crate::context::{TCContext, TCResult, TCValue};
use crate::error;
use crate::host::HostContext;
use crate::transaction::Transaction;

#[derive(Deserialize, Serialize)]
struct Op {
    context: String,
    method: String,
    args: HashMap<String, TCValue>,
}

#[derive(Deserialize, Serialize)]
enum PostRequest {
    Op(Op),
    Val(String, TCValue),
}

pub async fn listen(
    host: Arc<HostContext>,
    port: u16,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let make_svc = make_service_fn(|_conn| {
        let host = host.clone();
        async { Ok::<_, Infallible>(service_fn(move |req| handle(host.clone(), req))) }
    });

    let addr = ([127, 0, 0, 1], port).into();
    let server = Server::bind(&addr).serve(make_svc);

    println!("Listening on http://{}", addr);

    if let Err(e) = server.await {
        eprintln!("server error: {}", e);
    }

    Ok(())
}

async fn handle(
    host: Arc<HostContext>,
    req: Request<Body>,
) -> Result<Response<Body>, hyper::Error> {
    match *req.method() {
        Method::POST => {
            let context = req.uri().path().to_string();
            let params: HashMap<String, String> = req
                .uri()
                .query()
                .map(|v| {
                    url::form_urlencoded::parse(v.as_bytes())
                        .into_owned()
                        .collect()
                })
                .unwrap_or_else(HashMap::new);

            let capture = if let Some(param) = params.get("capture") {
                param.split('/').collect()
            } else {
                vec![]
            };

            let body = &hyper::body::to_bytes(req.into_body()).await?;
            let result = match serde_json::from_slice::<Vec<PostRequest>>(body) {
                Ok(requests) => match post(host.transaction(context), requests, capture).await {
                    Ok(values) => {
                        serde_json::to_string_pretty(&values)
                            .and_then(|s| Ok(s.into_bytes()))
                            .or_else(|e| {
                                let msg = "Your request completed successfully but there was an error serializing the response";
                                Err(error::bad_request(msg, e))
                            })
                    },
                    Err(cause) => Err(cause)
                },
                Err(cause) => {
                    Err(error::bad_request("Unable to parse request", cause))
                }
            };
            transform_error(result)
        }
        _ => {
            let mut response = Response::new(Body::from(""));
            *response.status_mut() = StatusCode::NOT_IMPLEMENTED;
            Ok(response)
        }
    }
}

async fn post(
    txn: Arc<Transaction>,
    requests: Vec<PostRequest>,
    capture: Vec<&str>,
) -> TCResult<HashMap<String, TCValue>> {
    for request in requests {
        let txn = txn.clone();
        match request {
            PostRequest::Val(name, value) => txn.provide(name, value)?,
            PostRequest::Op(op) => {
                let child_txn = txn.extend(op.context);
                child_txn.post(op.method, op.args)?;
            }
        }
    }

    Ok(txn.resolve(capture).await?)
}

fn transform_error(result: TCResult<Vec<u8>>) -> Result<Response<Body>, hyper::Error> {
    match result {
        Ok(contents) => Ok(Response::new(Body::from(contents))),
        Err(cause) => {
            let mut response = Response::new(Body::from(cause.message().to_string()));
            *response.status_mut() = match cause.reason() {
                error::Code::BadRequest => StatusCode::BAD_REQUEST,
                error::Code::MethodNotAllowed => StatusCode::METHOD_NOT_ALLOWED,
                error::Code::NotImplemented => StatusCode::NOT_IMPLEMENTED,
            };
            Ok(response)
        }
    }
}
