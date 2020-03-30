use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;

use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Method, Request, Response, Server, StatusCode};

use crate::context::{TCContext, TCOp, TCRequest, TCResult, TCValue};
use crate::error;
use crate::host::HostContext;
use crate::transaction::Transaction;

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
            let path = req.uri().path().to_string();
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
            let txn = host.clone().transaction();

            let result = if path == "/" {
                match serde_json::from_slice::<HashMap<String, TCRequest>>(body) {
                    Ok(requests) => transact(txn, requests, capture).await,
                    Err(cause) => {
                        return transform_error(Err(error::bad_request(
                            "Unable to parse request",
                            cause,
                        )));
                    }
                }
            } else {
                match serde_json::from_slice::<TCOp>(body) {
                    Ok(_op) => host.get(path).await.and_then(|c| match c {
                        _ => Err(error::not_implemented()),
                    }),
                    Err(cause) => Err(error::bad_request("Unable to parse request", cause)),
                }
            };

            let result = result.and_then(|values| {
                serde_json::to_string_pretty(&values)
                .and_then(|s| Ok(s.into_bytes()))
                .or_else(|e| {
                    let msg = "Your request completed successfully but there was an error serializing the response";
                    Err(error::bad_request(msg, e))
                })
            });

            transform_error(result)
        }
        _ => {
            let mut response = Response::new(Body::from(""));
            *response.status_mut() = StatusCode::NOT_IMPLEMENTED;
            Ok(response)
        }
    }
}

async fn transact(
    txn: Arc<Transaction>,
    requests: HashMap<String, TCRequest>,
    capture: Vec<&str>,
) -> TCResult<HashMap<String, TCValue>> {
    for (name, request) in requests {
        let txn = txn.clone();
        match request {
            TCRequest::Value(value) => txn.provide(name, value)?,
            TCRequest::Op(context, op) => {
                txn.extend(name, context, op);
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
                error::Code::NotFound => StatusCode::NOT_FOUND,
                error::Code::NotImplemented => StatusCode::NOT_IMPLEMENTED,
            };
            Ok(response)
        }
    }
}
