//! A [`TxnLock`] to support transaction-specific versioning

use std::collections::hash_map::{Entry, HashMap};
use std::collections::BTreeSet;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use async_trait::async_trait;
use futures::{join, TryFutureExt};
use log::{debug, trace};
use tokio::sync::broadcast::{self, Sender};
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock, RwLockWriteGuard};

use tc_error::*;

use crate::{Transact, TxnId, MIN_ID};

#[derive(Copy, Clone)]
struct Wake;

pub struct TxnLockReadGuard<T> {
    lock: TxnLock<T>,
    guard: OwnedRwLockReadGuard<T>,
}

impl<T> TxnLockReadGuard<T> {
    fn new(lock: TxnLock<T>, guard: OwnedRwLockReadGuard<T>) -> Self {
        Self { lock, guard }
    }
}

impl<T> Deref for TxnLockReadGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guard.deref()
    }
}

impl<T> Drop for TxnLockReadGuard<T> {
    fn drop(&mut self) {
        self.lock.wake();
    }
}

pub struct TxnLockWriteGuard<T> {
    lock: TxnLock<T>,
    guard: OwnedRwLockWriteGuard<T>,
}

impl<T> TxnLockWriteGuard<T> {
    fn new(lock: TxnLock<T>, guard: OwnedRwLockWriteGuard<T>) -> Self {
        Self { lock, guard }
    }
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

impl<T> Drop for TxnLockWriteGuard<T> {
    fn drop(&mut self) {
        self.lock.wake();
    }
}

struct Inner<T> {
    name: String,
    canon: RwLock<T>,
    versions: RwLock<HashMap<TxnId, Arc<RwLock<T>>>>,
    latest_read: RwLock<Option<TxnId>>,
    pending_writes: RwLock<BTreeSet<TxnId>>,
    last_commit: RwLock<Option<TxnId>>,
    tx: Sender<Wake>,
}

#[derive(Clone)]
pub struct TxnLock<T> {
    inner: Arc<Inner<T>>,
}

impl<T> TxnLock<T> {
    /// Create a new transactional lock.
    pub fn new<I: fmt::Display>(name: I, canon: T) -> Self {
        let (tx, _) = broadcast::channel(16);

        Self {
            inner: Arc::new(Inner {
                name: name.to_string(),
                canon: RwLock::new(canon),
                versions: RwLock::new(HashMap::new()),
                latest_read: RwLock::new(None),
                pending_writes: RwLock::new(BTreeSet::new()),
                last_commit: RwLock::new(None),
                tx,
            }),
        }
    }

    fn has_write_pending(
        pending_writes: &BTreeSet<TxnId>,
        txn_id: &TxnId,
        last_commit: &Option<TxnId>,
    ) -> bool {
        let last_commit = last_commit.unwrap_or(MIN_ID);
        for pending in pending_writes.iter().rev() {
            if pending < &last_commit {
                break;
            }

            if pending != txn_id {
                return true;
            }
        }

        false
    }

    async fn await_pending(
        &self,
        txn_id: &TxnId,
    ) -> TCResult<RwLockWriteGuard<'_, BTreeSet<TxnId>>> {
        debug!(
            "reserve version of {} at {} for reading",
            self.inner.name, txn_id
        );

        {
            let (last_commit, pending_writes) = join!(
                self.inner.last_commit.read(),
                self.inner.pending_writes.write()
            );

            if let Some(commit_id) = &*last_commit {
                if txn_id < commit_id {
                    return Err(TCError::conflict(format!(
                        "can't reserve {} at {} because the last commit is at {}",
                        self.inner.name, txn_id, commit_id
                    )));
                }
            }

            if !Self::has_write_pending(&pending_writes, txn_id, &*last_commit) {
                debug_assert!(pending_writes.iter().all(|pending| pending <= txn_id));
                return Ok(pending_writes);
            }
        }

        let mut rx = self.inner.tx.subscribe();
        loop {
            trace!("awaiting wakeup for lock {}...", self.inner.name);
            rx.recv().map_err(TCError::internal).await?;

            let (last_commit, pending_writes) = join!(
                self.inner.last_commit.read(),
                self.inner.pending_writes.write()
            );

            if let Some(commit_id) = &*last_commit {
                if txn_id < commit_id {
                    return Err(TCError::conflict(format!(
                        "can't reserve {} at {} because the last commit is at {}",
                        self.inner.name, txn_id, commit_id
                    )));
                }
            }

            if !Self::has_write_pending(&pending_writes, txn_id, &*last_commit) {
                debug_assert!(pending_writes.iter().all(|pending| pending <= txn_id));
                return Ok(pending_writes);
            }
        }
    }

    fn wake(&self) -> usize {
        match self.inner.tx.send(Wake) {
            Err(broadcast::error::SendError(_)) => 0,
            Ok(num_subscribed) => num_subscribed,
        }
    }
}

impl<T: Clone> TxnLock<T> {
    /// Try to acquire a read lock.
    pub async fn read(&self, txn_id: TxnId) -> TCResult<TxnLockReadGuard<T>> {
        debug!("read {} at {}", self.inner.name, txn_id);

        // if there's already a version at this TxnId, it's always safe to read
        if let Some(version) = {
            trace!("lock {} version collection", self.inner.name);
            let versions = self.inner.versions.read().await;
            versions.get(&txn_id).cloned()
        } {
            trace!(
                "there's already a version of {} at {}",
                self.inner.name,
                txn_id
            );

            let guard = version.clone().read_owned().await;
            trace!("locked {} for reading at {}", self.inner.name, txn_id);
            return Ok(TxnLockReadGuard::new(self.clone(), guard));
        }

        // await any pending writes which this version depends on
        let _pending_writes = self.await_pending(&txn_id).await?;

        trace!(
            "creating a new version of {} at {}...",
            self.inner.name,
            txn_id
        );

        let mut latest_read = self.inner.latest_read.write().await;
        if let Some(latest) = &mut *latest_read {
            if txn_id > *latest {
                *latest = txn_id;
            }
        }

        trace!("reading canonical version of {}", self.inner.name);
        let version = self.inner.canon.read().await;

        trace!("creating new version of {} at {}", self.inner.name, txn_id);
        let mut versions = self.inner.versions.write().await;
        let version = Arc::new(RwLock::new(version.deref().clone()));
        let guard = version.clone().read_owned().await;

        versions.insert(txn_id, version);

        trace!("created new version of {} at {}", self.inner.name, txn_id);
        Ok(TxnLockReadGuard::new(self.clone(), guard))
    }

    /// Try to acquire a write lock.
    pub async fn write(&self, txn_id: TxnId) -> TCResult<TxnLockWriteGuard<T>> {
        debug!("write {} at {}", self.inner.name, txn_id);

        let mut pending_writes = self.await_pending(&txn_id).await?;

        let latest_read = self.inner.latest_read.read().await;
        if let Some(latest_read) = &*latest_read {
            if latest_read > &txn_id {
                return Err(TCError::conflict(format!(
                    "can't lock {} for writing at {} since it's already been locked at {}",
                    self.inner.name, txn_id, latest_read
                )));
            }
        }

        trace!(
            "checking if {} is available for writing at {}...",
            self.inner.name,
            txn_id
        );

        trace!("getting the version of {} at {}", self.inner.name, txn_id);
        let version = {
            let mut versions = self.inner.versions.write().await;

            match versions.entry(txn_id) {
                Entry::Occupied(entry) => entry.get().clone(),
                Entry::Vacant(entry) => {
                    assert!(!pending_writes.contains(&txn_id));

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

        pending_writes.insert(txn_id);

        trace!(
            "lock existing version of {} for writing at {}",
            self.inner.name,
            txn_id
        );

        let guard = version.write_owned().await;
        Ok(TxnLockWriteGuard::new(self.clone(), guard))
    }
}

#[async_trait]
impl<T: Eq + Clone + Send + Sync> Transact for TxnLock<T> {
    async fn commit(&self, txn_id: &TxnId) {
        debug!("TxnLock::commit {} at {}", self.inner.name, txn_id);

        {
            let (mut canon, mut pending_writes, mut last_commit, latest_read) = join!(
                self.inner.canon.write(),
                self.inner.pending_writes.write(),
                self.inner.last_commit.write(),
                self.inner.latest_read.read()
            );

            let (updated, version) = {
                let mut versions = self.inner.versions.write().await;
                if let Some(version) = versions.get(txn_id) {
                    let version = version.read().await;

                    if pending_writes.contains(txn_id) {
                        (&*version != &*canon, version.clone())
                    } else {
                        assert!(&*version == &*canon);
                        (false, version.clone())
                    }
                } else {
                    // it's valid to request a read lock for the first time between commit & finalize
                    // so make sure to keep this version around
                    let version = canon.clone();
                    versions.insert(*txn_id, Arc::new(RwLock::new(version.clone())));
                    (false, version)
                }
            };

            for pending in pending_writes.iter() {
                if pending < txn_id {
                    // there's a write pending in the past
                    // if committing it would change the locked value, then it will crash on commit
                    // (this is handled by the code below)
                } else if pending > txn_id {
                    // there's a write pending in the future
                    // so make sure no changes were made in this version
                    if updated {
                        panic!(
                            "attempted to commit an out-of-order write to {} at {}",
                            self.inner.name, txn_id
                        );
                    }
                }
            }

            if updated {
                let has_been_read = if let Some(reserved) = &*latest_read {
                    reserved > txn_id
                } else {
                    false
                };

                if has_been_read {
                    if let Some(last_commit) = &*last_commit {
                        assert!(txn_id > &last_commit);
                    }

                    *last_commit = Some(*txn_id);
                }

                *canon = version.clone();
            }

            pending_writes.remove(txn_id);
        }

        self.wake();
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("finalize {} at {}", self.inner.name, txn_id);

        {
            let (mut pending_writes, mut versions) = join!(
                self.inner.pending_writes.write(),
                self.inner.versions.write()
            );
            pending_writes.remove(txn_id);
            versions.remove(txn_id);
        }

        self.wake();
    }
}
