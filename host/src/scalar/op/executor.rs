//! An executor for an `OpDef`

use std::collections::{HashMap, HashSet};

use futures::future::FutureExt;
use futures::stream::{FuturesUnordered, StreamExt};
use log::debug;

use error::*;
use generic::{Id, Instance, Map};

use crate::route::Public;
use crate::scalar::{Refer, Scope};
use crate::state::State;
use crate::txn::Txn;

/// An `OpDef` executor.
pub struct Executor<T> {
    txn: Txn,
    scope: Scope<T>,
}

impl<T: Clone + Instance + Public> Executor<T> {
    /// Construct a new `Executor` with the given [`Txn`] context and initial state.
    pub fn new<S: Into<State>, I: IntoIterator<Item = (Id, S)>>(
        txn: Txn,
        subject: T,
        data: I,
    ) -> Self {
        let scope = Scope::new(subject, data);
        Self { txn, scope }
    }

    pub fn with_context<S: Into<State>, I: IntoIterator<Item = (Id, S)>>(
        txn: Txn,
        subject: T,
        context: Map<State>,
        iter: I,
    ) -> Self {
        let scope = Scope::with_context(subject, context, iter);
        Self { txn, scope }
    }

    /// Resolve the state of the variable `capture`, including any of its dependencies.
    pub async fn capture(mut self, capture: Id) -> TCResult<State> {
        debug!("execute op & capture {}", capture);

        while self.scope.resolve_id(&capture)?.is_ref() {
            let mut visited = HashSet::with_capacity(self.scope.len());
            let mut pending = Vec::with_capacity(self.scope.len());
            let mut unvisited = Vec::with_capacity(self.scope.len());
            unvisited.push(capture.clone());

            while let Some(id) = unvisited.pop() {
                if visited.contains(&id) {
                    return Err(TCError::bad_request("circular dependency detected", id));
                } else {
                    visited.insert(id.clone());
                }

                let state = self.scope.resolve_id(&id)?;
                debug!("checking state {}: {}", id, state);

                if state.is_ref() {
                    let mut deps = HashSet::new();
                    state.requires(&mut deps);

                    let mut ready = true;
                    for dep_id in deps.into_iter() {
                        if self.scope.resolve_id(&dep_id)?.is_ref() {
                            ready = false;
                            unvisited.push(dep_id);
                        }
                    }

                    if ready {
                        pending.push(id);
                    } else {
                        debug!("{} still has unresolved deps", id);
                    }
                } else {
                    debug!("{} already resolved: {}", id, state);
                }
            }

            if pending.is_empty() && self.scope.resolve_id(&capture)?.is_ref() {
                return Err(TCError::bad_request(
                    "Cannot resolve all dependencies of",
                    capture,
                ));
            }

            let mut resolved = HashMap::with_capacity(pending.len());
            {
                let mut providers = FuturesUnordered::new();
                for id in pending.into_iter() {
                    let state = self.scope.resolve_id(&id)?.clone();
                    debug!("{} resolved to {}", id, state);
                    providers.push(state.resolve(&self.scope, &self.txn).map(|r| (id, r)));
                }

                while let Some((id, r)) = providers.next().await {
                    match r {
                        Ok(state) => {
                            resolved.insert(id, state);
                        }
                        Err(cause) => return Err(cause.consume(format!("Error resolving {}", id))),
                    }
                }
            }

            self.scope.extend(resolved);
        }

        self.scope
            .into_inner()
            .remove(&capture)
            .ok_or_else(|| TCError::not_found(capture))
    }
}
