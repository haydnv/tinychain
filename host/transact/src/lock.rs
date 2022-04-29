//! A [`TxnLock`] to support transaction-specific versioning

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::TryFutureExt;
use log::{debug, trace};
use tokio::sync::broadcast::{self, Sender};
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use tc_error::*;

use crate::{Transact, TxnId, MIN_ID};

#[derive(Copy, Clone)]
struct Wake;

pub struct TxnLockReadGuard<T> {
    lock: TxnLock<T>,
    txn_id: TxnId,
    guard: OwnedRwLockReadGuard<T>,
}

impl<T> TxnLockReadGuard<T> {
    fn new(lock: TxnLock<T>, txn_id: TxnId, guard: OwnedRwLockReadGuard<T>) -> Self {
        Self {
            lock,
            txn_id,
            guard,
        }
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
        trace!("TxnLockReadGuard::drop {}", self.lock.inner.name);

        let mut lock_state = self
            .lock
            .inner
            .state
            .lock()
            .expect("TxnLockReadGuard::drop");

        assert_ne!(lock_state.writer, Some(self.txn_id));

        let num_readers = lock_state
            .readers
            .get_mut(&self.txn_id)
            .expect("read lock count");

        *num_readers -= 1;

        trace!(
            "TxnLockReadGuard::drop {} has {} readers remaining",
            self.lock.inner.name,
            num_readers
        );

        if num_readers == &0 {
            self.lock.wake();
        }
    }
}

pub struct TxnLockWriteGuard<T> {
    lock: TxnLock<T>,
    txn_id: TxnId,
    guard: OwnedRwLockWriteGuard<T>,
}

impl<T> TxnLockWriteGuard<T> {
    fn new(lock: TxnLock<T>, txn_id: TxnId, guard: OwnedRwLockWriteGuard<T>) -> Self {
        Self {
            lock,
            txn_id,
            guard,
        }
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
        let mut state = self
            .lock
            .inner
            .state
            .lock()
            .expect("TxnLockWriteGuard::deref");
        state.pending_writes.insert(self.txn_id);

        self.guard.deref_mut()
    }
}

impl<T> Drop for TxnLockWriteGuard<T> {
    fn drop(&mut self) {
        trace!("TxnLockWriteGuard::drop {}", self.lock.inner.name);

        let mut lock_state = self
            .lock
            .inner
            .state
            .lock()
            .expect("TxnLockWriteGuard::drop");
        if let Some(readers) = lock_state.readers.get(&self.txn_id) {
            assert_eq!(readers, &0);
        }

        lock_state.writer = None;
        self.lock.wake();
    }
}

struct Versions<T> {
    canon: T,
    versions: HashMap<TxnId, Arc<RwLock<T>>>,
}

impl<T: Clone> Versions<T> {
    fn get(self: &mut Versions<T>, txn_id: TxnId) -> Arc<RwLock<T>> {
        if let Some(version) = self.versions.get(&txn_id) {
            version.clone()
        } else {
            let version = self.canon.clone();
            let version = Arc::new(RwLock::new(version));
            self.versions.insert(txn_id, version.clone());
            version
        }
    }
}

struct LockState<T> {
    versions: Versions<T>,
    readers: BTreeMap<TxnId, usize>,
    writer: Option<TxnId>,
    pending_writes: BTreeSet<TxnId>,
    last_commit: TxnId,
}

impl<T> LockState<T> {
    fn try_read(&mut self, txn_id: &TxnId) -> TCResult<Option<&mut Versions<T>>> {
        for reserved in self.pending_writes.iter().rev() {
            assert!(reserved <= &txn_id);

            if reserved > &self.last_commit && reserved < &txn_id {
                // if there's a pending write that can change the value at this txn_id, wait it out
                debug!("TxnLock waiting on a pending write at {}", reserved);
                return Ok(None);
            }
        }

        if self.writer.as_ref() == Some(txn_id) {
            // if there's an active writer for this txn_id, wait it out
            debug!("TxnLock waiting on a write lock at {}", txn_id);
            return Ok(None);
        }

        if !self.versions.versions.contains_key(&txn_id) {
            if txn_id <= &self.last_commit {
                // if the requested time is too old, just return an error
                return Err(TCError::conflict(format!(
                    "transaction {} is already finalized, can't acquire read lock",
                    txn_id
                )));
            }
        }

        *self.readers.entry(*txn_id).or_insert(0) += 1;

        Ok(Some(&mut self.versions))
    }

    fn try_write(&mut self, txn_id: &TxnId) -> TCResult<Option<&mut Versions<T>>> {
        if &self.last_commit >= txn_id {
            // can't write-lock a committed version
            return Err(TCError::conflict(format!(
                "can't acquire write lock at {} because of a commit at {}",
                txn_id, self.last_commit
            )));
        }

        for pending in self.pending_writes.iter().rev() {
            if pending > &txn_id {
                // can't write-lock the past
                return Err(TCError::conflict(format!(
                    "can't write at {} since there's already a lock at {}",
                    txn_id, pending
                )));
            } else if pending > &self.last_commit && pending < &txn_id {
                // if there's a past write that might still be committed, wait it out
                debug!(
                    "can't acquire write lock due to a pending write at {}",
                    pending
                );
                return Ok(None);
            }
        }

        if let Some(reader) = self.readers.keys().max() {
            if reader > &txn_id {
                // can't write-lock the past
                return Err(TCError::conflict(format!(
                    "can't acquire write lock at {} since it already has a read lock at {}",
                    txn_id, reader
                )));
            }
        }

        if let Some(readers) = self.readers.get(&txn_id) {
            // if there's an active read lock for this txn_id, wait it out
            if readers > &0 {
                debug!("TxnLock has {} active readers at {}", readers, txn_id);
                return Ok(None);
            }
        }

        if let Some(writer) = &self.writer {
            // if there's an active write lock, wait it out
            debug!("TxnLock has an active write lock at {}", writer);
            return Ok(None);
        }

        self.writer = Some(*txn_id);

        Ok(Some(&mut self.versions))
    }
}

impl<T: Clone + Eq> LockState<T> {
    fn commit(&mut self, txn_id: &TxnId) {
        if let Some(version) = self.versions.versions.get(txn_id) {
            let version = version.try_read().expect("transaction version");
            if version.deref() == &self.versions.canon {
                // no-op
            } else {
                self.last_commit = *txn_id;
                self.versions.canon = version.clone();
            }
        } else {
            // it's still valid to read the version at this transaction, so keep a copy around
            let canon = self.versions.canon.clone();
            self.versions
                .versions
                .insert(*txn_id, Arc::new(RwLock::new(canon)));
        };

        self.pending_writes.remove(txn_id);
    }

    fn finalize(&mut self, txn_id: &TxnId) {
        self.versions.versions.remove(txn_id);
        self.pending_writes.remove(txn_id);
    }
}

struct Inner<T> {
    name: String,
    state: Mutex<LockState<T>>,
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

        let versions = Versions {
            canon,
            versions: HashMap::new(),
        };

        let state = LockState {
            versions,
            readers: BTreeMap::new(),
            writer: None,
            pending_writes: BTreeSet::new(),
            last_commit: MIN_ID,
        };

        Self {
            inner: Arc::new(Inner {
                name: name.to_string(),
                state: Mutex::new(state),
                tx,
            }),
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

        let version = {
            let mut rx = self.inner.tx.subscribe();
            loop {
                {
                    let mut lock_state = self.inner.state.lock().expect("TxnLock::await_readable");
                    if let Some(versions) = lock_state.try_read(&txn_id)? {
                        break versions.get(txn_id);
                    }
                }

                let _updated = rx.recv().map_err(TCError::internal).await?;
            }
        };

        let guard = version.read_owned().await;
        Ok(TxnLockReadGuard::new(self.clone(), txn_id, guard))
    }

    /// Try to acquire a write lock.
    pub async fn write(&self, txn_id: TxnId) -> TCResult<TxnLockWriteGuard<T>> {
        debug!("write {} at {}", self.inner.name, txn_id);

        let version = {
            let mut rx = self.inner.tx.subscribe();
            loop {
                {
                    let mut lock_state = self.inner.state.lock().expect("TxnLock::await_writable");
                    if let Some(versions) = lock_state.try_write(&txn_id)? {
                        break versions.get(txn_id);
                    };
                }

                rx.recv().map_err(TCError::internal).await?;
            }
        };

        let guard = version.write_owned().await;
        Ok(TxnLockWriteGuard::new(self.clone(), txn_id, guard))
    }
}

#[async_trait]
impl<T: Eq + Clone + Send + Sync> Transact for TxnLock<T> {
    async fn commit(&self, txn_id: &TxnId) {
        debug!("TxnLock::commit {} at {}", self.inner.name, txn_id);

        {
            let mut lock_state = self.inner.state.lock().expect("TxnLock::commit");
            lock_state.commit(txn_id);
        }

        self.wake();
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("finalize {} at {}", self.inner.name, txn_id);

        {
            let mut lock_state = self.inner.state.lock().expect("TxnLock::finalize");
            lock_state.finalize(txn_id);
        }

        self.wake();
    }
}
