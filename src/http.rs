use std::convert::Infallible;
use std::sync::Arc;

use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Method, Request, Response, Server, StatusCode};

use crate::context::*;
use crate::error;
use crate::host::HostContext;

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
            let result = match req.uri().path().to_string().as_str() {
                "/" => {
                    let _txn = host.transaction();
                    Err(error::not_implemented())
                }
                _ => Err(error::not_implemented()),
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

fn transform_error(result: TCResult<TCValue>) -> Result<Response<Body>, hyper::Error> {
    match result {
		Ok(value) => {
			match serde_json::to_string_pretty(&value) {
				Ok(s) => Ok(Response::new(Body::from(s))),
				Err(e) => transform_error(Err(error::bad_request("Your request completed successfully but there was an error serializing the response:", e)))
			}
		},
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
