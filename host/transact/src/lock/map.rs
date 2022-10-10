//! A [`TxnMapLock`] to support transaction-specific versioning of a collection of states.

use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{Arc, Mutex};

use log::{debug, warn};
use tokio::sync::broadcast::{self, Sender};
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use tc_error::*;
use tcgeneric::{Id, Map};

use crate::TxnId;

use super::{Versions, Wake};

pub struct TxnMapLockReadGuard<T> {
    lock: TxnMapLock<T>,
    txn_id: TxnId,
    guard: OwnedRwLockReadGuard<BTreeSet<Id>>,
}

impl<T> TxnMapLockReadGuard<T> {
    pub fn get<K: Borrow<Id>>(&self, txn_id: TxnId, key: &Id) -> TCResult<Option<T>> {
        unimplemented!()
    }
}

pub struct TxnMapLockWriteGuard<T> {
    lock: TxnMapLock<T>,
    txn_id: TxnId,
    guard: OwnedRwLockWriteGuard<BTreeSet<Id>>,
}

impl<T> TxnMapLockWriteGuard<T> {
    pub fn get<K: Borrow<Id>>(&self, key: &Id) -> TCResult<Option<T>> {
        unimplemented!()
    }

    pub fn insert(&self, key: Id, value: T) -> TCResult<bool> {
        unimplemented!()
    }

    pub fn remove(&self, key: Id) -> TCResult<Option<T>> {
        unimplemented!()
    }
}

struct LockState<T> {
    keys: Versions<BTreeSet<Id>>,
    values: BTreeMap<Id, Versions<T>>,
}

impl<T> LockState<T> {
    fn try_read(&mut self, txn_id: &TxnId) -> TCResult<Option<Arc<RwLock<BTreeSet<Id>>>>> {
        unimplemented!()
    }

    fn try_write(&mut self, txn_id: &TxnId) -> TCResult<Option<Arc<RwLock<BTreeSet<Id>>>>> {
        unimplemented!()
    }
}

struct Inner<T> {
    name: String,
    state: Mutex<LockState<T>>,
    tx: Sender<Wake>,
}

#[derive(Clone)]
struct TxnMapLock<T> {
    inner: Arc<Inner<T>>,
}

impl<T> TxnMapLock<T> {
    pub fn new<I: fmt::Display>(name: I) -> Self {
        let (tx, _) = broadcast::channel(16);

        Self {
            inner: Arc::new(Inner {
                name: name.to_string(),
                state: Mutex::new(LockState {
                    keys: Versions::new(BTreeSet::new()),
                    values: BTreeMap::new(),
                }),
                tx,
            }),
        }
    }

    pub fn with_contents<I: fmt::Display>(name: I, contents: Map<T>) -> Self {
        let (tx, _) = broadcast::channel(16);

        let mut keys = BTreeSet::new();
        let mut values = BTreeMap::new();
        for (k, v) in contents.into_iter() {
            keys.insert(k.clone());
            values.insert(k, Versions::new(v));
        }

        Self {
            inner: Arc::new(Inner {
                name: name.to_string(),
                state: Mutex::new(LockState {
                    keys: Versions::new(keys),
                    values,
                }),
                tx,
            }),
        }
    }
}

impl<T: Clone> TxnMapLock<T> {
    /// Lock this map for reading at the given `txn_id`.
    pub async fn read(&self, txn_id: TxnId) -> TCResult<TxnMapLockReadGuard<T>> {
        debug!("lock map {} to read at {}...", self.inner.name, txn_id);

        let version = {
            let mut rx = self.inner.tx.subscribe();
            loop {
                {
                    let mut lock_state = self.inner.state.lock().expect("TxnMapLock state to read");
                    if let Some(version) = lock_state.try_read(&txn_id)? {
                        break version;
                    }
                }

                if let Err(cause) = rx.recv().await {
                    warn!("TxnMapLock wake error: {}", cause);
                }
            }
        };

        let guard = version.read_owned().await;
        let guard = TxnMapLockReadGuard {
            lock: self.clone(),
            txn_id,
            guard,
        };

        debug!("locked map {} for reading at {}", self.inner.name, txn_id);
        Ok(guard)
    }

    pub async fn write(&self, txn_id: TxnId) -> TCResult<TxnMapLockWriteGuard<T>> {
        debug!(
            "locking map {} for writing at {}...",
            self.inner.name, txn_id
        );

        let version = {
            let mut rx = self.inner.tx.subscribe();
            loop {
                {
                    let mut lock_state = self.inner.state.lock().expect("TxnMapLock state to write");
                    if let Some(version) = lock_state.try_write(&txn_id)? {
                        break version;
                    };
                }

                if let Err(cause) = rx.recv().await {
                    warn!("TxnMapLock wake error: {}", cause);
                }
            }
        };

        let guard = version.write_owned().await;
        let guard = TxnMapLockWriteGuard {
            lock: self.clone(),
            txn_id,
            guard,
        };

        debug!("locked {} for writing at {}", self.inner.name, txn_id);
        Ok(guard)
    }
}
