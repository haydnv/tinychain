//! A [`TxnLock`] to support transaction-specific versioning

use std::collections::hash_map::{Entry, HashMap};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use async_trait::async_trait;
use futures::join;
use log::{debug, trace};
use tokio::sync::{Mutex, OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use tc_error::*;

use crate::{Transact, TxnId};

pub struct TxnLockReadGuard<T> {
    guard: OwnedRwLockReadGuard<T>,
}

impl<T> Deref for TxnLockReadGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guard.deref()
    }
}

pub struct TxnLockWriteGuard<T> {
    guard: OwnedRwLockWriteGuard<T>,
}

impl<T> Deref for TxnLockWriteGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guard.deref()
    }
}

impl<T> DerefMut for TxnLockWriteGuard<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard.deref_mut()
    }
}

struct Inner<T> {
    name: String,
    canon: RwLock<T>,
    versions: Mutex<HashMap<TxnId, Arc<RwLock<T>>>>,
    reserved: Arc<RwLock<Option<TxnId>>>,
    last_commit: RwLock<Option<TxnId>>,
}

#[derive(Clone)]
pub struct TxnLock<T> {
    inner: Arc<Inner<T>>,
}

impl<T: Clone> TxnLock<T> {
    /// Create a new transactional lock.
    pub fn new<I: fmt::Display>(name: I, canon: T) -> Self {
        Self {
            inner: Arc::new(Inner {
                name: name.to_string(),
                canon: RwLock::new(canon),
                versions: Mutex::new(HashMap::new()),
                reserved: Arc::new(RwLock::new(None)),
                last_commit: RwLock::new(None),
            }),
        }
    }

    /// Try to acquire a read lock.
    pub async fn read(&self, txn_id: TxnId) -> TCResult<TxnLockReadGuard<T>> {
        debug!("read {} at {}", self.inner.name, txn_id);

        if let Some(version) = {
            trace!("lock {} version collection", self.inner.name);
            let versions = self.inner.versions.lock().await;
            versions.get(&txn_id).cloned()
        } {
            trace!(
                "there's already a version of {} at {}",
                self.inner.name,
                txn_id
            );

            // if there's already a version at this TxnId, it's always fine to read
            let guard = version.clone().read_owned().await;
            trace!("locked {} for reading at {}", self.inner.name, txn_id);
            return Ok(TxnLockReadGuard { guard });
            // the code below is responsible for making sure that no new version is created
            // in the case that there's a write pending
        }

        trace!(
            "creating a new version of {} at {}...",
            self.inner.name,
            txn_id
        );

        let reserved = self.inner.reserved.read().await;
        if let Some(reserved) = reserved.deref() {
            debug!("{} is reserved at {}", self.inner.name, reserved);

            if reserved > &txn_id {
                return Err(TCError::conflict(format!(
                    "cannot lock {} for reading at {} since it's already locked for writing at {}",
                    self.inner.name, txn_id, reserved
                )));
            } else {
                trace!(
                    "TxnLock::read checking the last commit ID for {}",
                    self.inner.name
                );

                if let Some(last_commit) = &*self.inner.last_commit.read().await {
                    if reserved > last_commit && reserved < &txn_id {
                        return Err(TCError::conflict(format!(
                            "cannot lock {} for reading at {} since it has a pending write at {}",
                            self.inner.name, txn_id, reserved
                        )));
                    }
                }
            }
        }

        trace!("reading canonical version of {}", self.inner.name);
        let version = self.inner.canon.read().await;

        trace!("creating new version of {} at {}", self.inner.name, txn_id);
        let mut versions = self.inner.versions.lock().await;
        let version = Arc::new(RwLock::new(version.deref().clone()));
        let guard = version.clone().read_owned().await;

        versions.insert(txn_id, version);

        trace!("created new version of {} at {}", self.inner.name, txn_id);
        Ok(TxnLockReadGuard { guard })
    }

    /// Try to acquire a write lock.
    pub async fn write(&self, txn_id: TxnId) -> TCResult<TxnLockWriteGuard<T>> {
        debug!("write {} at {}", self.inner.name, txn_id);

        let mut reserved = self.inner.reserved.clone().write_owned().await;
        let last_commit = self.inner.last_commit.read().await;

        // allow acquiring multiple write locks in sequence within a single transaction
        trace!("check if {} is reserved for writing...", self.inner.name);

        if let Some(reserved) = &*reserved {
            trace!("{} is reserved at {}", self.inner.name, reserved);

            if reserved == &txn_id {
                trace!(
                    "look up version of {} at {} for writing",
                    self.inner.name,
                    txn_id
                );

                let version = {
                    let versions = self.inner.versions.lock().await;
                    versions.get(&txn_id).expect("version").clone()
                };

                trace!("getting next version of {} at {}", self.inner.name, txn_id);
                let guard = version.clone().write_owned().await;
                trace!("got write lock on {} at {}", self.inner.name, txn_id);
                return Ok(TxnLockWriteGuard { guard });
            } else if let Some(last_commit) = &*last_commit {
                if reserved > last_commit {
                    return Err(TCError::conflict(format!(
                        "cannot write-lock {} at {} because it already has a pending write at {}",
                        self.inner.name, txn_id, reserved
                    )));
                } else if &txn_id <= last_commit {
                    return Err(TCError::conflict(format!(
                        "cannot lock {} for writing at {} because it was already committed at {}",
                        self.inner.name, txn_id, last_commit
                    )));
                }
            }
        }

        trace!("reserving {} for writing at {}...", self.inner.name, txn_id);
        *reserved = Some(txn_id);

        trace!(
            "TxnLock::write checking the last commit ID for {}",
            self.inner.name
        );

        trace!("getting the version of {} at {}", self.inner.name, txn_id);
        let version = {
            let mut versions = self.inner.versions.lock().await;

            match versions.entry(txn_id) {
                Entry::Occupied(entry) => entry.get().clone(),
                Entry::Vacant(entry) => {
                    trace!(
                        "create new version of {} for {} and lock for writing",
                        self.inner.name,
                        txn_id
                    );

                    let version = self.inner.canon.read().await;
                    let version = Arc::new(RwLock::new(version.deref().clone()));
                    entry.insert(version.clone());
                    version
                }
            }
        };

        trace!(
            "lock existing version of {} for writing at {}",
            self.inner.name,
            txn_id
        );

        let guard = version.write_owned().await;
        Ok(TxnLockWriteGuard { guard })
    }
}

#[async_trait]
impl<T: Clone + Send + Sync> Transact for TxnLock<T> {
    async fn commit(&self, txn_id: &TxnId) {
        debug!("commit {} at {}!", self.inner.name, txn_id);
        let (mut commit_id, mut canon) =
            join!(self.inner.last_commit.write(), self.inner.canon.write());

        trace!("prepared to commit {} at {}", self.inner.name, txn_id);

        if let Some(last_commit) = &*commit_id {
            assert!(txn_id > last_commit);
        }

        if let Some(version) = {
            let versions = self.inner.versions.lock().await;
            versions.get(txn_id).cloned()
        } {
            trace!(
                "TxnLock::commit {} reading version {}",
                self.inner.name,
                txn_id
            );

            let version = version.read().await;
            *canon = version.clone();
        }

        *commit_id = Some(*txn_id);

        debug!("committed {} committed at {}", self.inner.name, txn_id);
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("finalize {} at {}", self.inner.name, txn_id);

        let mut versions = self.inner.versions.lock().await;
        versions.remove(txn_id);
    }
}
