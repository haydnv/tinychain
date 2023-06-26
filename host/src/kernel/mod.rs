//! The host [`Kernel`], responsible for dispatching requests to the local host

use std::fmt;

use async_trait::async_trait;
use tc_error::*;
use tc_transact::TxnId;
use tc_value::Value;
use tcgeneric::*;

use crate::state::State;
use crate::txn::Txn;

use system::System;
use userspace::UserSpace;

pub use userspace::{Class, Library, Service, CLASS, LIB, SERVICE};

mod system;
mod userspace;

/// A part of the host [`Kernel`], responsible for dispatching requests to the local host
#[async_trait]
pub trait Dispatch {
    /// Route a GET request.
    async fn get(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<State>;

    /// Route a PUT request.
    async fn put(&self, txn: &Txn, path: &[PathSegment], key: Value, value: State) -> TCResult<()>;

    /// Route a POST request.
    async fn post(&self, txn: &Txn, path: &[PathSegment], data: State) -> TCResult<State>;

    /// Route a DELETE request.
    async fn delete(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<()>;

    /// Finalize a transaction
    async fn finalize(&self, txn_id: TxnId);
}

/// The host kernel, responsible for dispatching requests to the local host
pub struct Kernel {
    system: System,
    userspace: Option<UserSpace>,
}

impl Kernel {
    /// Initialize a new [`Kernel`] with no userspace.
    pub fn bootstrap() -> Self {
        Self {
            system: System,
            userspace: None,
        }
    }

    /// Initialize a new [`Kernel`] with the given userspace.
    pub fn with_userspace(class: Class, library: Library, service: Service) -> Self {
        Self {
            system: System,
            userspace: Some(UserSpace::new(class, library, service)),
        }
    }
}

#[async_trait]
impl Dispatch for Kernel {
    async fn get(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<State> {
        if let Some(userspace) = &self.userspace {
            if userspace.handles(path) {
                return userspace.get(txn, path, key).await;
            }
        }

        self.system.get(txn, path, key).await
    }

    async fn put(&self, txn: &Txn, path: &[PathSegment], key: Value, value: State) -> TCResult<()> {
        if let Some(userspace) = &self.userspace {
            if userspace.handles(path) {
                return userspace.put(txn, path, key, value).await;
            }
        }

        self.system.put(txn, path, key, value).await
    }

    async fn post(&self, txn: &Txn, path: &[PathSegment], data: State) -> TCResult<State> {
        if let Some(userspace) = &self.userspace {
            if userspace.handles(path) {
                return userspace.post(txn, path, data).await;
            }
        }

        self.system.post(txn, path, data).await
    }

    async fn delete(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<()> {
        if let Some(userspace) = &self.userspace {
            if userspace.handles(path) {
                return userspace.delete(txn, path, key).await;
            }
        }

        self.system.delete(txn, path, key).await
    }

    async fn finalize(&self, txn_id: TxnId) {
        if let Some(userspace) = &self.userspace {
            userspace.finalize(txn_id).await
        }
    }
}

impl fmt::Debug for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("host kernel")
    }
}
