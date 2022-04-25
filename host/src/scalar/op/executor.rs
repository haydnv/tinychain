//! An executor for an `OpDef`

use std::collections::{HashMap, HashSet, VecDeque};

use futures::future::FutureExt;
use futures::stream::{FuturesUnordered, StreamExt};
use log::debug;

use tc_error::*;
use tcgeneric::{Id, Instance, Map, Tuple};

use crate::route::Public;
use crate::scalar::{Refer, Scope};
use crate::state::{State, ToState};
use crate::txn::Txn;

struct Queue {
    order: VecDeque<Id>,
    set: HashSet<Id>,
}

impl Queue {
    fn with_capacity(size: usize) -> Self {
        Self {
            order: VecDeque::with_capacity(size),
            set: HashSet::with_capacity(size),
        }
    }

    fn into_iter(self) -> impl Iterator<Item = Id> {
        self.order.into_iter()
    }

    fn is_empty(&self) -> bool {
        self.set.is_empty()
    }

    fn len(&self) -> usize {
        self.set.len()
    }

    fn push_back(&mut self, id: Id) {
        if !self.set.contains(&id) {
            self.set.insert(id.clone());
            self.order.push_back(id)
        }
    }

    fn pop_front(&mut self) -> Option<Id> {
        if let Some(id) = self.order.pop_front() {
            self.set.remove(&id);
            Some(id)
        } else {
            None
        }
    }
}

/// An `OpDef` executor.
pub struct Executor<'a, T> {
    txn: &'a Txn,
    scope: Scope<'a, T>,
}

impl<'a, T: ToState + Instance + Public> Executor<'a, T> {
    /// Construct a new `Executor` with the given [`Txn`] context and initial state.
    pub fn new<S: Into<State>, I: IntoIterator<Item = (Id, S)>>(
        txn: &'a Txn,
        subject: Option<&'a T>,
        data: I,
    ) -> Self {
        let scope = Scope::new(subject, data);
        Self { txn, scope }
    }

    pub fn with_context<S: Into<State>, I: IntoIterator<Item = (Id, S)>>(
        txn: &'a Txn,
        subject: Option<&'a T>,
        context: Map<State>,
        iter: I,
    ) -> Self {
        let scope = Scope::with_context(subject, context, iter);
        Self { txn, scope }
    }

    pub fn from_scope(txn: &'a Txn, scope: Scope<'a, T>) -> Self {
        Self { txn, scope }
    }

    /// Resolve the state of the variable `capture`, including any of its dependencies.
    pub async fn capture(mut self, capture: Id) -> TCResult<State> {
        debug!("execute op & capture {}", capture);

        while self.scope.resolve_id(&capture)?.is_ref() {
            let mut pending = Queue::with_capacity(self.scope.len());
            let mut unvisited = Queue::with_capacity(self.scope.len());
            unvisited.push_back(capture.clone());

            while let Some(id) = unvisited.pop_front() {
                let state = self.scope.resolve_id(&id)?;
                debug!("checking state {}: {}", id, state);

                if state.is_ref() {
                    let mut deps = HashSet::new();
                    state.requires(&mut deps);

                    if deps.contains(&id) {
                        return Err(TCError::bad_request("circular dependency", id));
                    } else if deps.contains(&capture) {
                        return Err(TCError::unsupported(format!(
                            "circular dependency between {} and {}",
                            capture, id
                        )));
                    }

                    let mut ready = true;
                    for dep_id in deps.into_iter() {
                        if self.scope.resolve_id(&dep_id)?.is_ref() {
                            ready = false;
                            unvisited.push_back(dep_id);
                        }
                    }

                    if ready {
                        debug!("all deps resolved for {}", state);
                        pending.push_back(id);
                    } else {
                        debug!("{} still has unresolved deps", id);
                    }
                } else {
                    debug!("{} already resolved: {}", id, state);
                }
            }

            if pending.is_empty() {
                return Err(TCError::bad_request(
                    "cannot resolve all dependencies of",
                    capture,
                ));
            }

            let mut resolved = HashMap::with_capacity(pending.len());
            {
                let mut providers = FuturesUnordered::new();
                for id in pending.into_iter() {
                    let state = self.scope.resolve_id(&id)?;
                    let provider = state
                        .clone()
                        .resolve(&self.scope, &self.txn)
                        .map(|r| (id, r));

                    providers.push(provider);
                }

                while let Some((id, r)) = providers.next().await {
                    match r {
                        Ok(state) => {
                            debug!("{} resolved to {}", id, state);
                            resolved.insert(id, state);
                        }
                        Err(cause) => return Err(cause.consume(format!("while resolving {}", id))),
                    }
                }
            }

            self.scope.extend(resolved);
        }

        self.scope.remove(&capture).ok_or_else(|| {
            TCError::not_found(format!(
                "captured state {} in context {}",
                capture,
                self.scope.keys().collect::<Tuple<&Id>>()
            ))
        })
    }
}
