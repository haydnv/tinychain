//! The host kernel, responsible for dispatching requests to the local host.

use std::convert::TryInto;
use std::pin::Pin;

use futures::Future;
use log::debug;
use safecast::TryCastFrom;

use tc_error::*;
use tc_generic::*;

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

            cluster.get(&txn, suffix, key).await
        } else if &path[0] == "error" && path.len() == 2 {
            let message = String::try_cast_from(key, |v| {
                TCError::bad_request("cannot cast into error message string from", v)
            })?;

            if let Some(err_type) = error_type(&path[1]) {
                Err(TCError::new(err_type, message))
            } else {
                Err(TCError::not_found(TCPath::from(path)))
            }
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
        value: State,
    ) -> TCResult<()> {
        nonempty_path(path)?;

        if let Some(class) = StateType::from_path(path) {
            Err(TCError::method_not_allowed(class))
        } else if let Some((suffix, cluster)) = self.hosted.get(path) {
            debug!(
                "PUT {}: {} <- {} to cluster {}",
                TCPath::from(suffix),
                key,
                value,
                cluster
            );

            execute(txn, cluster, |txn, cluster| async move {
                cluster.put(&txn, suffix, key, value).await
            })
            .await
        } else {
            Err(TCError::not_found(TCPath::from(path)))
        }
    }

    /// Route a POST request.
    pub async fn post(&self, txn: &Txn, path: &[PathSegment], data: State) -> TCResult<State> {
        nonempty_path(path)?;

        if let Some((suffix, cluster)) = self.hosted.get(path) {
            let params: Map<State> = data.try_into()?;

            debug!(
                "POST {}: {} to cluster {}",
                TCPath::from(suffix),
                params,
                cluster
            );

            return if suffix.is_empty() && params.is_empty() {
                // it's a "commit" instruction
                cluster.post(&txn, suffix, params).await
            } else {
                execute(txn, cluster, |txn, cluster| async move {
                    cluster.post(&txn, suffix, params).await
                })
                .await
            };
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

fn execute<
    'a,
    R: Send,
    Fut: Future<Output = TCResult<R>> + Send,
    F: FnOnce(Txn, &'a InstanceExt<Cluster>) -> Fut + Send + 'a,
>(
    txn: &'a Txn,
    cluster: &'a InstanceExt<Cluster>,
    handler: F,
) -> Pin<Box<dyn Future<Output = TCResult<R>> + Send + 'a>> {
    Box::pin(async move {
        if let Some(owner_link) = txn.owner() {
            let link = txn.link(cluster.path().to_vec().into());
            if txn.is_owner(cluster.path()) {
                debug!("{} owns this transaction, no need to notify", link);
            } else {
                txn.put(owner_link.clone(), Value::default(), link.into())
                    .await?;
            }

            handler(txn.clone(), cluster).await
        } else {
            // Claim and execute the transaction
            let txn = cluster.claim(&txn).await?;
            let state = handler(txn.clone(), cluster).await?;

            let owner = cluster.owner(txn.id()).await?;

            owner.commit(&txn).await?;
            cluster.commit(txn.id()).await;

            Ok(state)
        }
    })
}

#[inline]
fn nonempty_path(path: &[PathSegment]) -> TCResult<()> {
    if path.is_empty() {
        Err(TCError::method_not_allowed(TCPathBuf::default()))
    } else {
        Ok(())
    }
}

fn error_type(err_type: &Id) -> Option<ErrorType> {
    match err_type.as_str() {
        "bad_gateway" => Some(ErrorType::BadGateway),
        "bad_request" => Some(ErrorType::BadRequest),
        "conflict" => Some(ErrorType::Conflict),
        "forbidden" => Some(ErrorType::Forbidden),
        "internal" => Some(ErrorType::Internal),
        "method_not_allowed" => Some(ErrorType::MethodNotAllowed),
        "not_found" => Some(ErrorType::NotFound),
        "not_implemented" => Some(ErrorType::NotImplemented),
        "timeout" => Some(ErrorType::Timeout),
        "unauthorized" => Some(ErrorType::Unauthorized),
        _ => None,
    }
}
