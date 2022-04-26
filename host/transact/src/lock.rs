//! A [`TxnLock`] to support transaction-specific versioning

use std::collections::hash_map::{Entry, HashMap};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use async_trait::async_trait;
use futures::{join, TryFutureExt};
use log::{debug, trace};
use tokio::sync::broadcast::{self, Sender};
use tokio::sync::{
    Mutex, OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock, RwLockReadGuard, RwLockWriteGuard,
};

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
    latest_read: Arc<RwLock<Option<TxnId>>>,
    pending_write: Arc<RwLock<Option<TxnId>>>,
    last_commit: RwLock<Option<TxnId>>,
    tx: Sender<TxnId>,
}

#[derive(Clone)]
pub struct TxnLock<T> {
    inner: Arc<Inner<T>>,
}

impl<T: Clone> TxnLock<T> {
    /// Create a new transactional lock.
    pub fn new<I: fmt::Display>(name: I, canon: T) -> Self {
        let (tx, _) = broadcast::channel(16);

        Self {
            inner: Arc::new(Inner {
                name: name.to_string(),
                canon: RwLock::new(canon),
                versions: Mutex::new(HashMap::new()),
                latest_read: Arc::new(RwLock::new(None)),
                pending_write: Arc::new(RwLock::new(None)),
                last_commit: RwLock::new(None),
                tx,
            }),
        }
    }

    async fn reserve_read(&self, txn_id: &TxnId) -> TCResult<RwLockReadGuard<'_, Option<TxnId>>> {
        debug!(
            "reserve version of {} at {} for reading",
            self.inner.name, txn_id
        );

        {
            let pending = self.inner.pending_write.read().await;
            if let Some(pending_id) = &*pending {
                if pending_id >= txn_id {
                    return Ok(pending);
                } else {
                    debug!(
                        "can't read {} at {} since there is a write pending at {}",
                        self.inner.name, txn_id, pending_id
                    );
                }
            } else {
                return Ok(pending);
            }
        }

        let mut rx = self.inner.tx.subscribe();
        loop {
            trace!("awaiting commit of {}...", self.inner.name);
            let commit_id = rx.recv().map_err(TCError::internal).await?;

            if &commit_id >= txn_id {
                return Err(TCError::conflict(format!(
                    "past version {} of {} is no longer available for reading",
                    txn_id, self.inner.name
                )));
            }

            let pending = self.inner.pending_write.read().await;
            if let Some(pending_id) = &*pending {
                if pending_id >= txn_id {
                    return Ok(pending);
                } else {
                    debug!("{} has a pending write at {}", self.inner.name, pending_id);
                }
            } else {
                return Ok(pending);
            }
        }
    }

    /// Try to acquire a read lock.
    pub async fn read(&self, txn_id: TxnId) -> TCResult<TxnLockReadGuard<T>> {
        debug!("read {} at {}", self.inner.name, txn_id);

        // if there's already a version at this TxnId, it's always safe to read
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

            let guard = version.clone().read_owned().await;
            trace!("locked {} for reading at {}", self.inner.name, txn_id);
            return Ok(TxnLockReadGuard { guard });
        }

        // await any pending writes which this version depends on
        self.reserve_read(&txn_id).await?;

        trace!(
            "creating a new version of {} at {}...",
            self.inner.name,
            txn_id
        );

        let mut latest_read = self.inner.latest_read.write().await;
        if let Some(latest) = &mut *latest_read {
            if *latest < txn_id {
                *latest = txn_id;
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

    async fn reserve_write(&self, txn_id: &TxnId) -> TCResult<RwLockWriteGuard<'_, Option<TxnId>>> {
        debug!(
            "reserve version of {} at {} for writing",
            self.inner.name, txn_id
        );

        {
            let reserved = self.inner.pending_write.write().await;
            if let Some(pending) = &*reserved {
                if pending == txn_id {
                    return Ok(reserved);
                } else {
                    debug!(
                        "{} has a pending write at {}, can't lock at {}",
                        self.inner.name, pending, txn_id
                    );
                }
            } else {
                return Ok(reserved);
            }
        }

        let mut rx = self.inner.tx.subscribe();
        loop {
            let commit_id = rx.recv().map_err(TCError::internal).await?;

            if &commit_id > txn_id {
                return Err(TCError::conflict(format!(
                    "cannot lock {} for writing at {} because it's already committed at {}",
                    self.inner.name, txn_id, commit_id
                )));
            }

            let reserved = self.inner.pending_write.write().await;
            if let Some(pending) = &*reserved {
                if pending == txn_id {
                    return Ok(reserved);
                } else {
                    debug!("{} has a pending write at {}", self.inner.name, pending);
                }
            } else {
                return Ok(reserved);
            }
        }
    }

    /// Try to acquire a write lock.
    pub async fn write(&self, txn_id: TxnId) -> TCResult<TxnLockWriteGuard<T>> {
        debug!("write {} at {}", self.inner.name, txn_id);

        let mut reserved = self.reserve_write(&txn_id).await?;

        if let Some(reserved) = &*reserved {
            trace!("{} is reserved at {}", self.inner.name, reserved);
            assert_eq!(reserved, &txn_id);
        }

        *reserved = Some(txn_id);

        trace!(
            "checking if {} is available for writing at {}...",
            self.inner.name,
            txn_id
        );

        let last_commit = self.inner.last_commit.read().await;
        if let Some(last_commit) = &*last_commit {
            if last_commit > &txn_id {
                return Err(TCError::conflict(format!(
                    "cannot lock {} for writing at {} because it was already committed at {}",
                    self.inner.name, txn_id, last_commit
                )));
            }
        }

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
impl<T: Eq + Clone + Send + Sync> Transact for TxnLock<T> {
    async fn commit(&self, txn_id: &TxnId) {
        debug!("commit {} at {}!", self.inner.name, txn_id);
        let (mut commit_id, mut canon, mut reserved) = join!(
            self.inner.last_commit.write(),
            self.inner.canon.write(),
            self.inner.pending_write.write()
        );

        trace!("prepared to commit {} at {}", self.inner.name, txn_id);

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
            if &*version != &*canon {
                *canon = version.clone();

                if let Some(last_commit) = &*commit_id {
                    assert!(txn_id > last_commit);
                }
            }
        }

        *commit_id = Some(*txn_id);
        *reserved = None;

        let subscribed = match self.inner.tx.send(*txn_id) {
            Err(broadcast::error::SendError(_)) => 0,
            Ok(num_subscribed) => num_subscribed,
        };

        debug!(
            "committed {} at {}; {} subscribers",
            self.inner.name, txn_id, subscribed
        );
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("finalize {} at {}", self.inner.name, txn_id);

        let mut versions = self.inner.versions.lock().await;
        versions.remove(txn_id);
    }
}
