use async_trait::async_trait;

use crate::auth;
use crate::class::{State, TCType};
use crate::error;
use crate::general::Map;
use crate::request::Request;
use crate::scalar::{MethodType, PathSegment, Scalar, Value};
use crate::transaction::Txn;
use crate::TCResult;

#[async_trait]
pub trait Handler: Send + Sync {
    fn subject(&self) -> TCType;

    fn scope(&self) -> Option<auth::Scope> {
        None
    }

    fn authorize(&self, _request: &Request) -> TCResult<()> {
        if let Some(_scope) = self.scope() {
            // TODO: validate scope
        }

        Ok(())
    }

    async fn handle_get(self: Box<Self>, _txn: &Txn, _key: Value) -> TCResult<State> {
        Err(error::method_not_allowed(self.subject()))
    }

    async fn get(self: Box<Self>, request: &Request, txn: &Txn, key: Value) -> TCResult<State> {
        self.authorize(request)?;
        self.handle_get(txn, key).await
    }

    async fn handle_put(
        self: Box<Self>,
        _request: &Request,
        _txn: &Txn,
        _key: Value,
        _value: State,
    ) -> TCResult<()> {
        Err(error::method_not_allowed(self.subject()))
    }

    async fn put(
        self: Box<Self>,
        request: &Request,
        txn: &Txn,
        key: Value,
        value: State,
    ) -> TCResult<()> {
        self.authorize(request)?;
        self.handle_put(request, txn, key, value).await
    }

    // TODO: params: Map<State>
    async fn handle_post(
        self: Box<Self>,
        _request: &Request,
        _txn: &Txn,
        _params: Map<Scalar>,
    ) -> TCResult<State> {
        Err(error::method_not_allowed(self.subject()))
    }

    async fn post(
        self: Box<Self>,
        request: &Request,
        txn: &Txn,
        params: Map<Scalar>,
    ) -> TCResult<State> {
        self.authorize(request)?;
        self.handle_post(request, txn, params).await
    }

    async fn handle_delete(self: Box<Self>, _txn: &Txn, _key: Value) -> TCResult<()> {
        Err(error::method_not_allowed(self.subject()))
    }

    async fn delete(self: Box<Self>, request: &Request, txn: &Txn, key: Value) -> TCResult<()> {
        self.authorize(request)?;
        self.handle_delete(txn, key).await
    }
}

#[async_trait]
pub trait Public {
    async fn get(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        key: Value,
    ) -> TCResult<State>;

    async fn put(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        key: Value,
        value: State,
    ) -> TCResult<()>;

    async fn post(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        params: Map<Scalar>,
    ) -> TCResult<State>;

    async fn delete(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        key: Value,
    ) -> TCResult<()>;
}

pub trait Route {
    fn route(
        &'_ self,
        method: MethodType,
        path: &'_ [PathSegment],
    ) -> Option<Box<dyn Handler + '_>>;
}

#[async_trait]
impl<T: Route + Send + Sync> Public for T {
    async fn get(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        key: Value,
    ) -> TCResult<State> {
        if let Some(handler) = self.route(MethodType::Get, path) {
            handler.get(request, txn, key).await
        } else {
            Err(error::path_not_found(path))
        }
    }

    async fn put(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        key: Value,
        value: State,
    ) -> TCResult<()> {
        if let Some(handler) = self.route(MethodType::Put, path) {
            handler.put(request, txn, key, value).await
        } else {
            Err(error::path_not_found(path))
        }
    }

    async fn post(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        params: Map<Scalar>,
    ) -> TCResult<State> {
        if let Some(handler) = self.route(MethodType::Post, path) {
            handler.post(request, txn, params).await
        } else {
            Err(error::path_not_found(path))
        }
    }

    async fn delete(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        key: Value,
    ) -> TCResult<()> {
        if let Some(handler) = self.route(MethodType::Delete, path) {
            handler.delete(request, txn, key).await
        } else {
            Err(error::path_not_found(path))
        }
    }
}
