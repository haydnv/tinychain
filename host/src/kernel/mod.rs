//! The host [`Kernel`], responsible for dispatching requests to the local host

use std::fmt;

use async_trait::async_trait;
use tc_error::*;
use tc_value::Value;
use tcgeneric::*;

use crate::cluster::{Cluster, Legacy};
use crate::object::InstanceExt;
use crate::state::State;
use crate::txn::{hypothetical, Txn};

use crate::txn::hypothetical::Hypothetical;
use hosted::Hosted;
use system::System;
use userspace::UserSpace;

pub use userspace::{Library, LIB};

mod hosted; // TODO: delete
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
}

/// The host kernel, responsible for dispatching requests to the local host
pub struct Kernel {
    system: System,
    userspace: Option<UserSpace>,
}

impl Kernel {
    /// Initialize a new [`Kernel`] with no [`UserSpace`].
    pub fn bootstrap() -> Self {
        Self {
            system: System,
            userspace: None,
        }
    }

    /// Initialize a new [`Kernel`] with no [`UserSpace`].
    pub fn with_userspace<I>(library: Library, clusters: I) -> Self
    where
        I: IntoIterator<Item = InstanceExt<Cluster<Legacy>>>,
    {
        Self {
            system: System,
            userspace: Some(UserSpace::new(library, clusters)),
        }
    }

    // TODO: delete
    pub fn hosted(&self) -> Box<dyn Iterator<Item = &InstanceExt<Cluster<Legacy>>> + '_> {
        if let Some(userspace) = &self.userspace {
            Box::new(userspace.hosted())
        } else {
            Box::new(std::iter::empty())
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
}

impl fmt::Display for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("host kernel")
    }
}
