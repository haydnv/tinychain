use async_trait::async_trait;

use crate::auth;
use crate::class::{State, TCResult, TCType};
use crate::error;
use crate::request::Request;
use crate::scalar::{Object, PathSegment, Value};
use crate::transaction::Txn;

#[async_trait]
pub trait Handler: Send + Sync {
    fn subject(&self) -> TCType;

    fn scope(&self) -> auth::Scope;

    async fn handle_get(&self, _txn: &Txn, _key: Value) -> TCResult<State> {
        Err(error::method_not_allowed(self.subject()))
    }

    async fn get(&self, _request: &Request, txn: &Txn, key: Value) -> TCResult<State> {
        // TODO: validate scope
        self.handle_get(txn, key).await
    }

    async fn handle_put(&self, _txn: &Txn, _key: Value, _value: State) -> TCResult<()> {
        Err(error::method_not_allowed(self.subject()))
    }

    async fn put(&self, _request: &Request, txn: &Txn, key: Value, value: State) -> TCResult<()> {
        // TODO: validate scope
        self.handle_put(txn, key, value).await
    }

    async fn handle_post(&self, _txn: &Txn, _params: Object) -> TCResult<State> {
        Err(error::method_not_allowed(self.subject()))
    }

    async fn post(&self, _request: &Request, txn: &Txn, params: Object) -> TCResult<State> {
        // TODO: validate scope
        self.handle_post(txn, params).await
    }

    async fn handle_delete(&self, _txn: &Txn, _key: Value) -> TCResult<()> {
        Err(error::method_not_allowed(self.subject()))
    }

    async fn delete(&self, _request: &Request, txn: &Txn, key: Value) -> TCResult<()> {
        // TODO: validate scope
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
