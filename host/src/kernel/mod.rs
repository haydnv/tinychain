//! The host kernel, responsible for dispatching requests to the local host.

use std::convert::TryInto;

use log::debug;

use error::*;
use generic::*;

use crate::cluster::Cluster;
use crate::object::InstanceExt;
use crate::route::Public;
use crate::scalar::*;
use crate::state::*;
use crate::txn::*;

mod hosted;

use hosted::Hosted;

/// A host kernel, responsible for dispatching requests to the local host.
pub struct Kernel {
    hosted: Hosted,
}

impl Kernel {
    /// Construct a new `Kernel` to host the given [`Cluster`]s.
    pub fn new<I: IntoIterator<Item = InstanceExt<Cluster>>>(clusters: I) -> Self {
        Self {
            hosted: clusters.into_iter().collect(),
        }
    }

    /// Route a GET request.
    pub async fn get(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<State> {
        nonempty_path(path)?;

        if let Some(class) = StateType::from_path(path) {
            let err = format!("Cannot cast into {} from {}", class, key);
            State::Scalar(Scalar::Value(key))
                .into_type(class)
                .ok_or_else(|| TCError::unsupported(err))
        } else if let Some((suffix, cluster)) = self.hosted.get(path) {
            debug!(
                "GET {}: {} from cluster {}",
                TCPath::from(suffix),
                key,
                cluster
            );

            txn.mutate((*cluster).clone()).await?;
            cluster.get(txn, suffix, key).await
        } else {
            Err(TCError::not_found(TCPath::from(path)))
        }
    }

    /// Route a PUT request.
    pub async fn put(
        &self,
        txn: &Txn,
        path: &[PathSegment],
        key: Value,
        state: State,
    ) -> TCResult<()> {
        nonempty_path(path)?;

        if let Some(class) = StateType::from_path(path) {
            Err(TCError::method_not_allowed(class))
        } else if let Some((suffix, cluster)) = self.hosted.get(path) {
            txn.mutate((*cluster).clone()).await?;
            cluster.put(txn, suffix, key, state).await
        } else {
            Err(TCError::not_found(TCPath::from(path)))
        }
    }

    /// Route a POST request.
    pub async fn post(&self, txn: &Txn, path: &[PathSegment], data: State) -> TCResult<State> {
        nonempty_path(path)?;

        if let Some((suffix, cluster)) = self.hosted.get(path) {
            let params = data.try_into()?;
            txn.mutate((*cluster).clone()).await?;
            return cluster.post(txn, suffix, params).await;
        }

        match path[0].as_str() {
            "transact" if path.len() == 1 => Err(TCError::method_not_allowed(path[0].as_str())),
            "transact" if path.len() == 2 => match path[1].as_str() {
                "execute" => Err(TCError::not_implemented("/transact/execute")),
                "hypothetical" => Err(TCError::not_implemented("hypothetical queries")),
                other => Err(TCError::not_found(other)),
            },
            other => Err(TCError::not_found(other)),
        }
    }
}

#[inline]
fn nonempty_path(path: &[PathSegment]) -> TCResult<()> {
    if path.is_empty() {
        Err(TCError::method_not_allowed(TCPathBuf::default()))
    } else {
        Ok(())
    }
}
