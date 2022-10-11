//! A [`TxnLock`] to support transaction-specific versioning

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use log::{debug, trace, warn};
use tokio::sync::broadcast::{self, Sender};
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use tc_error::*;

use crate::{Transact, TxnId, MIN_ID};

use super::{Versions, Wake};

pub struct TxnLockReadGuard<T: Clone + PartialEq> {
    lock: TxnLock<T>,
    txn_id: TxnId,
    guard: OwnedRwLockReadGuard<T>,
}

impl<T: Clone + PartialEq> TxnLockReadGuard<T> {
    #[inline]
    pub fn id(&self) -> &TxnId {
        &self.txn_id
    }
}

impl<T: Clone + PartialEq> Clone for TxnLockReadGuard<T> {
    fn clone(&self) -> Self {
        trace!("TxnLockReadGuard::clone {}", self.lock.inner.name);

        let mut lock_state = self
            .lock
            .inner
            .state
            .lock()
            .expect("TxnLockReadGuard::clone");

        let num_readers = lock_state
            .readers
            .get_mut(&self.txn_id)
            .expect("read lock count");

        *num_readers += 1;

        let guard = lock_state
            .versions
            .get(self.txn_id)
            .try_read_owned()
            .expect("transaction version read guard");

        Self {
            lock: self.lock.clone(),
            txn_id: self.txn_id,
            guard,
        }
    }
}

impl<T: Clone + PartialEq> Deref for TxnLockReadGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guard.deref()
    }
}

impl<T: Clone + PartialEq> Drop for TxnLockReadGuard<T> {
    fn drop(&mut self) {
        trace!("TxnLockReadGuard::drop {}", self.lock.inner.name);

        let num_readers = {
            let mut lock_state = self
                .lock
                .inner
                .state
                .lock()
                .expect("TxnLockReadGuard::drop");

            lock_state.drop_read(&self.txn_id, false)
        };

        if num_readers == 0 {
            self.lock.wake();
        }
    }
}

pub struct TxnLockReadGuardExclusive<T: Clone + PartialEq> {
    lock: TxnLock<T>,
    txn_id: TxnId,
    guard: OwnedRwLockWriteGuard<T>,
    pending_upgrade: bool,
}

impl<T: Clone + PartialEq> TxnLockReadGuardExclusive<T> {
    #[inline]
    pub fn id(&self) -> &TxnId {
        &self.txn_id
    }
}

impl<T: Clone + PartialEq> TxnLockReadGuardExclusive<T> {
    pub fn upgrade(mut self) -> TxnLockWriteGuard<T> {
        let lock = self.lock.clone();
        let txn_id = self.txn_id;

        self.pending_upgrade = true;
        std::mem::drop(self);

        lock.try_write(txn_id).expect("upgrade exclusive read lock")
    }
}

impl<T: Clone + PartialEq> Deref for TxnLockReadGuardExclusive<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl<T: Clone + PartialEq> Drop for TxnLockReadGuardExclusive<T> {
    fn drop(&mut self) {
        trace!("TxnLockReadGuardExclusive::drop {}", self.lock.inner.name);

        let num_readers = {
            let mut lock_state = self
                .lock
                .inner
                .state
                .lock()
                .expect("TxnLockReadGuardExclusive::drop");

            lock_state.drop_read(&self.txn_id, true)
        };

        if self.pending_upgrade {
            trace!("TxnLockReadGuardExclusive::drop pending upgrade, not waking subscribers...");
        } else if num_readers == 0 {
            trace!("TxnLockReadGuardExclusive::drop waking subscribers...");
            self.lock.wake();
        }
    }
}

pub struct TxnLockWriteGuard<T: Clone + PartialEq> {
    lock: TxnLock<T>,
    txn_id: TxnId,
    guard: OwnedRwLockWriteGuard<T>,
    pending_downgrade: bool,
}

impl<T: Clone + PartialEq> TxnLockWriteGuard<T> {
    #[inline]
    pub fn id(&self) -> &TxnId {
        &self.txn_id
    }
}

impl<T: Clone + PartialEq> TxnLockWriteGuard<T> {
    fn new(lock: TxnLock<T>, txn_id: TxnId, guard: OwnedRwLockWriteGuard<T>) -> Self {
        Self {
            lock,
            txn_id,
            guard,
            pending_downgrade: false,
        }
    }

    pub fn downgrade(mut self) -> TxnLockReadGuardExclusive<T> {
        let lock = self.lock.clone();
        let txn_id = self.txn_id;

        self.pending_downgrade = true;
        std::mem::drop(self);

        lock.try_read_exclusive(txn_id)
            .expect("downgrade write lock")
    }
}

impl<T: Clone + PartialEq> Deref for TxnLockWriteGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guard.deref()
    }
}

impl<T: Clone + PartialEq> DerefMut for TxnLockWriteGuard<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard.deref_mut()
    }
}

impl<T: Clone + PartialEq> Drop for TxnLockWriteGuard<T> {
    fn drop(&mut self) {
        trace!("TxnLockWriteGuard::drop {}", self.lock.inner.name);

        {
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
        }

        if !self.pending_downgrade {
            self.lock.wake();
        }
    }
}

struct LockState<T> {
    versions: Versions<T>,
    readers: BTreeMap<TxnId, usize>,
    exclusive: BTreeSet<TxnId>,
    writer: Option<TxnId>,
    pending_writes: BTreeSet<TxnId>,
    last_commit: TxnId,
}

impl<T: Clone + PartialEq> LockState<T> {
    fn drop_read(&mut self, txn_id: &TxnId, exclusive: bool) -> usize {
        if let Some(writer) = &self.writer {
            assert_ne!(writer, txn_id);
        }

        let num_readers = self.readers.get_mut(txn_id).expect("read lock count");
        *num_readers -= 1;

        trace!(
            "txn lock has {} readers remaining after dropping one read guard",
            num_readers
        );

        if exclusive {
            assert_eq!(*num_readers, 0);
            self.exclusive.remove(txn_id);
        }

        *num_readers
    }

    fn try_read(&mut self, txn_id: &TxnId, exclusive: bool) -> TCResult<Option<Arc<RwLock<T>>>> {
        if self.exclusive.contains(txn_id) {
            debug!("TxnLock is locked exclusively for reading");
            return Ok(None);
        }

        for reserved in self.pending_writes.iter().rev() {
            debug_assert!(reserved <= &txn_id);

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

        let num_readers = self.readers.entry(*txn_id).or_insert(0);

        if exclusive {
            if *num_readers == 0 {
                self.exclusive.insert(*txn_id);
            } else {
                trace!("TxnLock is locked non-exclusively for reading");
                return Ok(None);
            }
        }

        *num_readers += 1;

        Ok(Some(self.versions.get(*txn_id).clone()))
    }

    fn try_write(&mut self, txn_id: &TxnId) -> TCResult<Option<Arc<RwLock<T>>>> {
        if &self.last_commit >= txn_id {
            // can't write-lock a committed version
            return Err(TCError::conflict(format!(
                "can't acquire write lock at {} because of a commit at {}",
                txn_id, self.last_commit
            )));
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

        if let Some(writer) = &self.writer {
            assert!(writer <= txn_id);
            // if there's an active write lock, wait it out
            debug!("TxnLock has an active write lock at {}", writer);
            return Ok(None);
        } else if let Some(readers) = self.readers.get(&txn_id) {
            // if there's an active read lock for this txn_id, wait it out
            if readers > &0 {
                debug!("TxnLock has {} active readers at {}", readers, txn_id);
                return Ok(None);
            }
        }

        self.writer = Some(*txn_id);
        self.pending_writes.insert(*txn_id);

        Ok(Some(self.versions.get(*txn_id).clone()))
    }
}

impl<T: Clone + Eq> LockState<T> {
    fn commit(&mut self, txn_id: &TxnId) {
        if self.versions.commit(txn_id) {
            self.last_commit = *txn_id;
        }

        self.pending_writes.remove(txn_id);
    }

    fn finalize(&mut self, txn_id: &TxnId) {
        self.versions.finalize(txn_id);
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

        let versions = Versions::new(canon);

        let state = LockState {
            versions,
            readers: BTreeMap::new(),
            exclusive: BTreeSet::new(),
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
        trace!(
            "TxnLock {} waking {} subscribers",
            self.inner.name,
            self.inner.tx.receiver_count()
        );

        match self.inner.tx.send(Wake) {
            Err(broadcast::error::SendError(_)) => 0,
            Ok(num_subscribed) => {
                trace!(
                    "TxnLock {} woke {} subscribers",
                    self.inner.name,
                    num_subscribed
                );

                num_subscribed
            }
        }
    }
}

impl<T: Clone + PartialEq> TxnLock<T> {
    /// Lock this value for reading at the given `txn_id`.
    pub async fn read(&self, txn_id: TxnId) -> TCResult<TxnLockReadGuard<T>> {
        debug!("lock {} to read at {}...", self.inner.name, txn_id);

        let version = {
            let mut rx = self.inner.tx.subscribe();
            loop {
                {
                    let mut lock_state = self.inner.state.lock().expect("TxnLock state to read");
                    if let Some(version) = lock_state.try_read(&txn_id, false)? {
                        break version;
                    }
                }

                if let Err(cause) = rx.recv().await {
                    warn!("TxnLock wake error: {}", cause);
                }
            }
        };

        let guard = version.read_owned().await;
        let guard = TxnLockReadGuard {
            lock: self.clone(),
            txn_id,
            guard,
        };

        debug!("locked {} for reading at {}", self.inner.name, txn_id);
        Ok(guard)
    }

    /// Lock this value exclusively for reading at the given `txn_id`.
    pub async fn read_exclusive(&self, txn_id: TxnId) -> TCResult<TxnLockReadGuardExclusive<T>> {
        debug!(
            "lock {} exclusively to read at {}...",
            self.inner.name, txn_id
        );

        let version = {
            let mut rx = self.inner.tx.subscribe();
            loop {
                {
                    let mut lock_state = self
                        .inner
                        .state
                        .lock()
                        .expect("TxnLock state to read exclusively");

                    if let Some(version) = lock_state.try_read(&txn_id, true)? {
                        break version;
                    }
                }

                if let Err(cause) = rx.recv().await {
                    warn!("TxnLock wake error: {}", cause);
                }
            }
        };

        let guard = version.write_owned().await;
        let guard = TxnLockReadGuardExclusive {
            lock: self.clone(),
            txn_id,
            guard,
            pending_upgrade: false,
        };

        debug!("locked {} for reading at {}", self.inner.name, txn_id);
        Ok(guard)
    }

    /// Lock this value exclusively for reading at the given `txn_id`, if possible.
    pub fn try_read_exclusive(&self, txn_id: TxnId) -> TCResult<TxnLockReadGuardExclusive<T>> {
        debug!(
            "try to lock {} exclusively to read at {}...",
            self.inner.name, txn_id
        );

        const ERR: &str = "could not acquire transactional exclusive-read lock";

        let version = {
            let mut lock_state = self.inner.state.lock().expect("TxnLock state to write");
            if let Some(version) = lock_state.try_read(&txn_id, true)? {
                Ok(version)
            } else {
                Err(TCError::conflict(ERR))
            }
        }?;

        let guard = version
            .try_write_owned()
            .map_err(|cause| TCError::conflict(format!("{}: {}", ERR, cause)))?;

        let guard = TxnLockReadGuardExclusive {
            lock: self.clone(),
            txn_id,
            guard,
            pending_upgrade: false,
        };

        debug!("locked {} for reading at {}", self.inner.name, txn_id);
        Ok(guard)
    }

    /// Try to acquire a write lock synchronously, if possible.
    pub fn try_write(&self, txn_id: TxnId) -> TCResult<TxnLockWriteGuard<T>> {
        const ERR: &str = "could not acquire transactional read lock";

        let version = {
            let mut lock_state = self.inner.state.lock().expect("TxnLock::await_readable");
            if let Some(version) = lock_state.try_write(&txn_id)? {
                version
            } else {
                return Err(TCError::conflict(ERR));
            }
        };

        let guard = version
            .try_write_owned()
            .map_err(|cause| TCError::conflict(format!("{}: {}", ERR, cause)))?;

        Ok(TxnLockWriteGuard::new(self.clone(), txn_id, guard))
    }

    /// Lock this value for writing at the given `txn_id`.
    pub async fn write(&self, txn_id: TxnId) -> TCResult<TxnLockWriteGuard<T>> {
        debug!("locking {} for writing at {}...", self.inner.name, txn_id);

        let version = {
            let mut rx = self.inner.tx.subscribe();
            loop {
                {
                    let mut lock_state = self.inner.state.lock().expect("TxnLock::await_writable");
                    if let Some(version) = lock_state.try_write(&txn_id)? {
                        break version;
                    };
                }

                if let Err(cause) = rx.recv().await {
                    warn!("TxnLock wake error: {}", cause);
                }
            }
        };

        let guard = version.write_owned().await;
        let guard = TxnLockWriteGuard::new(self.clone(), txn_id, guard);

        debug!("locked {} for writing at {}", self.inner.name, txn_id);
        Ok(guard)
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
