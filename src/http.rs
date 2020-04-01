use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;

use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Method, Request, Response, Server, StatusCode};

use crate::context::{Link, TCResult, TCValue};
use crate::error;
use crate::host::Host;

pub async fn listen(
    host: Arc<Host>,
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

async fn handle(host: Arc<Host>, req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
    let path = match Link::to(req.uri().path()) {
        Ok(link) => link,
        Err(cause) => {
            return transform_error(Err(cause));
        }
    };

    let _params: HashMap<String, String> = req
        .uri()
        .query()
        .map(|v| {
            url::form_urlencoded::parse(v.as_bytes())
                .into_owned()
                .collect()
        })
        .unwrap_or_else(HashMap::new);

    match *req.method() {
        Method::POST => {
            let body = &hyper::body::to_bytes(req.into_body()).await?;
            let args = match serde_json::from_slice::<HashMap<String, TCValue>>(body) {
                Ok(args) => args,
                Err(cause) => {
                    return transform_error(Err(error::bad_request(
                        "Unable to parse request",
                        cause,
                    )));
                }
            };

            let txn = match host.clone().new_transaction() {
                Ok(txn) => txn,
                Err(cause) => {
                    return transform_error(Err(cause));
                }
            };

            let args = args
                .iter()
                .map(|(name, arg)| (name.as_str(), arg.clone()))
                .collect();

            let result = match txn.post(path, args).await {
                Ok(result) => match result.to_value() {
                    Ok(value) => value,
                    Err(cause) => {
                        let msg = "The request completed successfully but the result could not be transmitted via HTTP";
                        let err = error::TCError::of(
                            cause.reason().clone(),
                            format!("{}: {}", msg, cause.message()),
                        );
                        return transform_error(Err(err));
                    }
                },
                Err(cause) => {
                    return transform_error(Err(cause));
                }
            };

            let result = serde_json::to_string_pretty(&result)
                .and_then(|s| Ok(s.into_bytes()))
                .or_else(|e| {
                    let msg = "Your request completed successfully but there was an error serializing the response";
                    Err(error::bad_request(msg, e))
                });

            transform_error(result)
        }
        _ => {
            let mut response = Response::new(Body::from(""));
            *response.status_mut() = StatusCode::NOT_FOUND;
            Ok(response)
        }
    }
}

fn transform_error(result: TCResult<Vec<u8>>) -> Result<Response<Body>, hyper::Error> {
    match result {
        Ok(contents) => Ok(Response::new(Body::from(contents))),
        Err(cause) => {
            let mut response = Response::new(Body::from(cause.message().to_string()));
            *response.status_mut() = match cause.reason() {
                error::Code::BadRequest => StatusCode::BAD_REQUEST,
                error::Code::Internal => StatusCode::INTERNAL_SERVER_ERROR,
                error::Code::MethodNotAllowed => StatusCode::METHOD_NOT_ALLOWED,
                error::Code::NotFound => StatusCode::NOT_FOUND,
                error::Code::NotImplemented => StatusCode::NOT_IMPLEMENTED,
            };
            Ok(response)
        }
    }
}
