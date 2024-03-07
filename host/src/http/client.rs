use std::sync::Arc;

use async_trait::async_trait;
use destream::de;
use futures::{StreamExt, TryFutureExt, TryStreamExt};
use http_body_util::{BodyStream, Empty, StreamBody};
use hyper::body::{Body, Bytes, Frame, Incoming};
use hyper_util::rt::TokioIo;
use tokio::net::TcpStream;

use tc_error::*;
use tc_server::{
    Actor, IntoView, RPCClient, State, StateView, Transaction, Txn, TxnId, VerifyingKey,
};
use tc_value::{ToUrl, Value};
use tcgeneric::Map;

use super::Encoding;

const ERR_NO_OWNER: &str = "an ownerless transaction may not make outgoing requests";

/// A TinyChain HTTP client. Should only be used through a `Gateway`.
pub struct Client;

impl Client {
    /// Construct a new `Client`.
    pub fn new() -> Self {
        Self
    }

    async fn connect_and_send<B>(
        &self,
        stream: TcpStream,
        request: hyper::http::request::Builder,
        body: B,
    ) -> TCResult<hyper::Response<Incoming>>
    where
        B: Body<Data = Bytes> + Send + Unpin + 'static,
        B::Error: std::error::Error + Send + Sync + 'static,
    {
        let io = TokioIo::new(stream);

        // TCP handshake
        let (mut sender, conn) = hyper::client::conn::http1::handshake(io)
            .map_err(|cause| bad_gateway!("handshake failed").consume(cause))
            .await?;

        // check that the connection succeeded
        conn.map_err(|cause| bad_gateway!("connection refused").consume(cause))
            .await?;

        let request = request
            .body(body)
            .map_err(|cause| internal!("encoding outbound request failed").consume(cause))?;

        // TODO: implement a timeout
        sender
            .send_request(request)
            .map_err(|cause| bad_gateway!("{cause}"))
            .await
    }

    async fn request<T>(
        &self,
        method: &str,
        link: ToUrl<'_>,
        txn_id: &TxnId,
        key: &Value,
        value: Option<StateView<'static>>,
        auth: Option<String>,
        cxt: T::Context,
    ) -> TCResult<T>
    where
        T: de::FromStream,
    {
        let url = if key.is_some() {
            let key_json = serde_json::to_string(&key)
                .map_err(|cause| internal!("unable to encode key {}", key).consume(cause))?;

            let key_encoded =
                url::form_urlencoded::byte_serialize(key_json.as_bytes()).collect::<String>();

            format!("{link}?txn_id={txn_id}&key={key_encoded}")
        } else {
            format!("{link}?txn_id={txn_id}")
        };

        let url: hyper::Uri = url
            .parse()
            .map_err(|cause| bad_request!("invalid URI").consume(cause))?;

        let host = url
            .host()
            .ok_or_else(|| internal!("outgoing request is missing a hostname"))?;

        let port = url.port_u16().unwrap_or(80);
        let addr = format!("{}:{}", host, port);

        let request = hyper::Request::builder()
            .method(method)
            .header(hyper::header::HOST, host)
            .header(hyper::header::ACCEPT_ENCODING, Encoding::Tbon.as_str())
            .uri(url);

        let request = if let Some(token) = auth {
            request.header(hyper::header::AUTHORIZATION, format!("Bearer {}", token))
        } else {
            request
        };

        let stream = TcpStream::connect(addr)
            .map_err(|cause| bad_gateway!("connection refused").consume(cause))
            .await?;

        let response = if let Some(value) = value {
            let body = tbon::en::encode(value)
                .map_err(|cause| internal!("unable to encode outgoing request").consume(cause))?;

            let body = body
                .map_ok(Frame::data)
                .map_err(|cause| internal!("TBON encoding error").consume(cause));

            let body = StreamBody::new(body);

            self.connect_and_send(stream, request, body)
                .map_err(|cause| bad_gateway!("upstream error").consume(cause))
                .await?
        } else {
            let body = Empty::<Bytes>::new();

            self.connect_and_send(stream, request, body)
                .map_err(|cause| bad_gateway!("upstream error").consume(cause))
                .await?
        };

        if response.status().is_success() {
            let body = BodyStream::new(response.into_body())
                .map_ok(|frame| frame.into_data().expect("frame"));

            tbon::de::try_decode(cxt, body)
                .map_err(|cause| {
                    #[cfg(debug_assertions)]
                    log::warn!("upstream error: {cause}");

                    bad_gateway!("error decoding response from {}", link).consume(cause)
                })
                .await
        } else {
            let err = transform_error(&link, response).await;
            Err(bad_gateway!("error from upstream host").consume(err))
        }
    }
}

#[async_trait]
impl RPCClient for Client {
    async fn fetch(&self, txn_id: TxnId, link: ToUrl<'_>, actor_id: Value) -> TCResult<Actor> {
        let public_key: Value = self
            .request("GET", link, &txn_id, &actor_id, None, None, ())
            .await?;

        let public_key = Arc::<[u8]>::try_from(public_key)?;
        let public_key = VerifyingKey::try_from(&*public_key)
            .map_err(|cause| unauthorized!("invalid public key for {actor_id}: {cause}"))?;

        Ok(Actor::with_public_key(actor_id, public_key))
    }

    async fn get(&self, txn: &Txn, link: ToUrl<'_>, key: Value) -> TCResult<State> {
        let token = self.extract_jwt(txn);
        let txn_id = txn.id();
        let txn = txn.subcontext_unique();
        self.request("GET", link, txn_id, &key, None, token, txn)
            .await
    }

    async fn put(&self, txn: &Txn, link: ToUrl<'_>, key: Value, value: State) -> TCResult<()> {
        if txn.owner().map(|owner| owner.is_some())? {
            let value = value.into_view(txn.subcontext_unique()).await?;
            let token = self.extract_jwt(txn);
            self.request("PUT", link, txn.id(), &key, Some(value), token, ())
                .await
        } else {
            Err(bad_request!("{ERR_NO_OWNER}"))
        }
    }

    async fn post(&self, txn: &Txn, link: ToUrl<'_>, params: Map<State>) -> TCResult<State> {
        if txn.owner().map(|owner| owner.is_some())? {
            let params = State::Map(params);
            let params = params.into_view(txn.subcontext_unique()).await?;
            let token = self.extract_jwt(txn);
            let txn_id = txn.id();
            let txn = txn.subcontext_unique();

            self.request("POST", link, txn_id, &Value::None, Some(params), token, txn)
                .await
        } else {
            Err(bad_request!("{ERR_NO_OWNER}"))
        }
    }

    async fn delete(&self, txn: &Txn, link: ToUrl<'_>, key: Value) -> TCResult<()> {
        if txn.owner().map(|owner| owner.is_some())? {
            let token = self.extract_jwt(txn);

            self.request("DELETE", link, txn.id(), &key, None, token, ())
                .await
        } else {
            Err(bad_request!("{ERR_NO_OWNER}"))
        }
    }
}

async fn transform_error(source: &ToUrl<'_>, response: hyper::Response<Incoming>) -> TCError {
    const MAX_ERR_SIZE: usize = 5000;

    let status = response.status();

    let mut body = BodyStream::new(response.into_body());
    let mut err = Vec::new();
    loop {
        if err.len() >= MAX_ERR_SIZE {
            break;
        } else if let Some(Ok(buf)) = body.next().await {
            if buf.is_data() {
                err.extend(buf.into_data().expect("frame"));
            } else {
                break;
            }
        } else {
            break;
        }
    }

    let message = if let Ok(message) = String::from_utf8(err) {
        format!("error from upstream host {}: {}", source, message)
    } else {
        format!("error from upstream host {}", source)
    };

    use hyper::StatusCode;
    let code = match status {
        StatusCode::BAD_REQUEST => ErrorKind::BadRequest,
        StatusCode::CONFLICT => ErrorKind::Conflict,
        StatusCode::FORBIDDEN => ErrorKind::Forbidden,
        StatusCode::INTERNAL_SERVER_ERROR => ErrorKind::Internal,
        StatusCode::GATEWAY_TIMEOUT => ErrorKind::Timeout,
        StatusCode::METHOD_NOT_ALLOWED => ErrorKind::MethodNotAllowed,
        StatusCode::NOT_FOUND => ErrorKind::NotFound,
        StatusCode::NOT_IMPLEMENTED => ErrorKind::NotImplemented,
        StatusCode::UNAUTHORIZED => ErrorKind::Unauthorized,
        StatusCode::REQUEST_TIMEOUT => ErrorKind::Timeout,
        _ => ErrorKind::BadGateway,
    };

    TCError::new(code, message)
}
