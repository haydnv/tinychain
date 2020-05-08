use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;

use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Method, Request, Response, Server, StatusCode};

use crate::error;
use crate::host::Host;
use crate::state::State;
use crate::value::{Args, Op, TCPath, TCRef, TCResult, TCValue, ValueId};

const UNSERIALIZABLE: &str =
    "The request completed successfully but some of the response could not be serialized";

fn line_numbers(s: &str) -> String {
    s.lines()
        .enumerate()
        .map(|(i, l)| format!("{} {}", i, l))
        .collect::<Vec<String>>()
        .join("\n")
}

pub async fn listen(
    host: Arc<Host>,
    address: &SocketAddr,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let make_svc = make_service_fn(|_conn| {
        let host = host.clone();
        async { Ok::<_, Infallible>(service_fn(move |req| handle(host.clone(), req))) }
    });

    let server = Server::bind(address).serve(make_svc);

    println!("Listening on http://{}", address);

    if let Err(e) = server.await {
        eprintln!("server error: {}", e);
    }

    Ok(())
}

async fn get(host: Arc<Host>, path: &TCPath, key: TCValue) -> TCResult<State> {
    host.get(host.new_transaction()?, path, key).await
}

async fn post(host: Arc<Host>, path: &TCPath, mut args: Args) -> TCResult<State> {
    if path == "/sbin/transact" {
        let capture: Vec<ValueId> = args.take_or("capture", vec![])?;
        let values: Vec<(ValueId, TCValue)> = args.take_or("values", vec![])?;
        let txn = host
            .clone()
            .transact(Op::post(None, TCPath::default(), values))?;

        let mut results: Vec<TCValue> = Vec::with_capacity(capture.len());
        match txn.execute(capture.into_iter().collect()).await {
            Ok(responses) => {
                for (id, r) in responses {
                    match r {
                        State::Value(val) => {
                            results.push((TCRef::from(id), val).into());
                        }
                        other => {
                            return Err(error::bad_request(
                                "Attempt to capture an unserializable value",
                                other,
                            ));
                        }
                    }
                }
            }
            Err(cause) => {
                return Err(cause);
            }
        };

        txn.commit().await;

        Ok(State::Value(results.into()))
    } else {
        host.post(host.new_transaction()?, path, args).await
    }
}

async fn route(
    host: Arc<Host>,
    method: Method,
    path: &str,
    params: HashMap<String, String>,
    body: Vec<u8>,
) -> TCResult<Vec<u8>> {
    let path: TCPath = path.parse()?;

    match method {
        Method::GET => {
            let key = if let Some(key) = params.get("key") {
                serde_json::from_str::<TCValue>(key)
                    .map_err(|e| error::bad_request("Unable to parse 'key' param", e))?
            } else {
                TCValue::None
            };

            match get(host, &path, key).await? {
                State::Value(val) => Ok(serde_json::to_string_pretty(&val)?.as_bytes().to_vec()),
                state => Err(error::bad_request(
                    "Attempt to GET unserializable state {}",
                    state,
                )),
            }
        }
        Method::POST => {
            let args: HashMap<ValueId, TCValue> = match serde_json::from_slice(&body) {
                Ok(params) => params,
                Err(cause) => {
                    let body = line_numbers(std::str::from_utf8(&body).unwrap());
                    return Err(error::bad_request(
                        &format!("{}\n\nUnable to parse request", body),
                        cause,
                    ));
                }
            };

            match post(host, &path, args.into()).await? {
                State::Value(v) => serde_json::to_string_pretty(&v)
                    .and_then(|s| Ok(s.into_bytes()))
                    .or_else(|e| Err(error::bad_request(UNSERIALIZABLE, e))),
                other => Err(error::bad_request(UNSERIALIZABLE, other)),
            }
        }
        _ => Err(error::not_found(path)),
    }
}

async fn handle(host: Arc<Host>, req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let path = uri.path();

    let params: HashMap<String, String> = uri
        .query()
        .map(|v| {
            url::form_urlencoded::parse(v.as_bytes())
                .into_owned()
                .collect()
        })
        .unwrap_or_else(HashMap::new);

    let body = &hyper::body::to_bytes(req.into_body()).await?;

    transform_error(route(host, method, path, params, body.to_vec()).await)
}

fn transform_error(result: TCResult<Vec<u8>>) -> Result<Response<Body>, hyper::Error> {
    match result {
        Ok(contents) => Ok(Response::new(Body::from(contents))),
        Err(cause) => {
            let mut response = Response::new(Body::from(cause.message().to_string()));
            *response.status_mut() = match cause.reason() {
                error::Code::BadRequest => StatusCode::BAD_REQUEST,
                error::Code::Forbidden => StatusCode::FORBIDDEN,
                error::Code::Internal => StatusCode::INTERNAL_SERVER_ERROR,
                error::Code::MethodNotAllowed => StatusCode::METHOD_NOT_ALLOWED,
                error::Code::NotFound => StatusCode::NOT_FOUND,
                error::Code::NotImplemented => StatusCode::NOT_IMPLEMENTED,
                error::Code::Unauthorized => StatusCode::UNAUTHORIZED,
            };
            Ok(response)
        }
    }
}
