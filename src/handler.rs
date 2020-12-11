use async_trait::async_trait;

use crate::auth;
use crate::class::{State, TCResult};
use crate::request::Request;
use crate::scalar::{Object, PathSegment, Value};
use crate::transaction::Txn;

pub trait Handler {
    fn scope() -> auth::Scope;
}

#[async_trait]
pub trait HandleGet: Handler {
    async fn handle_get(txn: &Txn, key: Value) -> TCResult<State>;

    async fn get(_request: &Request, txn: &Txn, key: Value) -> TCResult<State> {
        Self::handle_get(txn, key).await
    }
}

#[async_trait]
pub trait HandlePut: Handler {
    async fn handle_put(txn: &Txn, key: Value, value: State) -> TCResult<State>;

    async fn put(_request: &Request, txn: &Txn, key: Value, value: State) -> TCResult<State> {
        // TODO: validate scope
        Self::handle_put(txn, key, value).await
    }
}

#[async_trait]
pub trait HandlePost: Handler {
    async fn handle_post(txn: &Txn, params: Object) -> TCResult<State>;

    async fn post(_request: &Request, txn: &Txn, params: Object) -> TCResult<State> {
        // TODO: validate scope
        Self::handle_post(txn, params).await
    }
}

#[async_trait]
pub trait HandleDelete: Handler {
    async fn handle_delete(txn: &Txn, key: Value) -> TCResult<State>;

    async fn delete(_request: &Request, txn: &Txn, key: Value) -> TCResult<State> {
        // TODO: validate scope
        Self::handle_delete(txn, key).await
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
        params: Object,
    ) -> TCResult<State>;

    async fn delete(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        key: Value,
    ) -> TCResult<()>;
}
