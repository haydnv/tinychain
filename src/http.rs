use std::convert::Infallible;
use std::sync::Arc;

use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, Server, StatusCode};

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
    _host: Arc<HostContext>,
    _req: Request<Body>,
) -> Result<Response<Body>, hyper::Error> {
    let mut response = Response::new(Body::from(""));
    *response.status_mut() = StatusCode::NOT_IMPLEMENTED;
    Ok(response)
}
