use std::collections::HashMap;
use std::convert::{Infallible, TryInto};
use std::net::SocketAddr;
use std::sync::Arc;

use hyper::header::HeaderValue;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Method, Request, Response, Server, StatusCode};

use crate::auth::Token;
use crate::error;
use crate::host::Host;
use crate::state::State;
use crate::transaction::Txn;
use crate::value::link::*;
use crate::value::op::Op;
use crate::value::{TCRef, TCResult, TCValue, ValueId};

const UNSERIALIZABLE: &str =
    "The request completed successfully but some of the response could not be serialized";

fn line_numbers(s: &str) -> String {
    s.lines()
        .enumerate()
        .map(|(i, l)| format!("{} {}", i + 1, l))
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

async fn get<'a>(
    txn: &'a Arc<Txn<'a>>,
    path: TCPath,
    key: TCValue,
    auth: &Option<Token>,
) -> TCResult<State> {
    txn.get(path.into(), key, auth).await
}

async fn post<'a>(
    txn: &'a Arc<Txn<'a>>,
    path: &TCPath,
    mut args: HashMap<ValueId, TCValue>,
    auth: &'a Option<Token>,
) -> TCResult<State> {
    if path == "/sbin/transact" {
        let capture: Vec<ValueId> = args
            .remove(&"capture".parse().unwrap())
            .map(|v| v.try_into())
            .unwrap_or_else(|| Ok(Vec::new()))?;
        let mut values: Vec<(ValueId, TCValue)> = args
            .remove(&"values".parse().unwrap())
            .map(|v| v.try_into())
            .unwrap_or_else(|| Ok(Vec::new()))?;
        txn.extend(values.drain(..), auth).await?;

        let mut results: Vec<TCValue> = Vec::with_capacity(capture.len());
        match txn.resolve(capture.into_iter().collect()).await {
            Ok(responses) => {
                for (id, r) in responses {
                    match r {
                        State::Value(val) => {
                            results.push((TCRef::from(id), val).into());
                        }
                        other => {
                            txn.rollback().await;
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
        Err(error::method_not_allowed(path))
    }
}

async fn route<'a>(
    txn: &'a Arc<Txn<'a>>,
    method: Method,
    path: &str,
    params: HashMap<String, String>,
    body: Vec<u8>,
    auth: &'a Option<Token>,
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

            match get(txn, path, key, &auth).await? {
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

            match post(txn, &path, args, auth).await? {
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

    let txn = match host.new_transaction().await {
        Ok(txn) => txn,
        Err(cause) => return transform_error(Err(cause)),
    };

    let token = if let Some(header) = req.headers().get("Authorization") {
        match validate_token(txn.clone(), header).await {
            Ok(token) => Some(token),
            Err(cause) => return transform_error(Err(cause)),
        }
    } else {
        None
    };

    let body = &hyper::body::to_bytes(req.into_body()).await?;

    transform_error(route(&txn, method, path, params, body.to_vec(), &token).await)
}

async fn validate_token(txn: Arc<Txn<'_>>, auth_header: &HeaderValue) -> TCResult<Token> {
    match auth_header.to_str() {
        Ok(t) => {
            if t.starts_with("Bearer: ") {
                let token: Token = t[8..].parse()?;
                let value_id: ValueId = "__actor_id".parse().unwrap();
                txn.push((value_id.clone(), Op::from(token.actor_id()).into()), &None)
                    .await?;
                txn.resolve(vec![value_id.clone()])
                    .await
                    .map(|_actor| Err(error::not_implemented()))?
            } else {
                Err(error::unauthorized(&format!(
                    "Invalid authorization header: {}",
                    t
                )))
            }
        }
        Err(cause) => Err(error::unauthorized(&cause.to_string())),
    }
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
