//! A [`TxnLock`] to support transaction-specific versioning

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use async_trait::async_trait;
use log::{debug, info, trace};
use tokio::sync::{Mutex, Notify, OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use tc_error::*;

use crate::{Transact, TxnId};

struct Versions<T> {
    canon: Arc<RwLock<T>>,
    versions: BTreeMap<TxnId, Arc<RwLock<T>>>,
    last_commit: TxnId,
}

impl<T> Versions<T> {
    fn new(canon: T) -> Self {
        Self {
            canon: Arc::new(RwLock::new(canon)),
            versions: BTreeMap::new(),
            last_commit: crate::id::MIN_ID,
        }
    }
}

impl<T: Clone> Versions<T> {
    async fn get(&mut self, txn_id: TxnId) -> TCResult<Arc<RwLock<T>>> {
        if let Some(version) = self.versions.get(&txn_id) {
            Ok(version.clone())
        } else {
            if txn_id < self.last_commit {
                return Err(TCError::conflict(format!(
                    "version {} has already been finalized",
                    txn_id
                )));
            }

            let canon = self.canon.read().await;
            let version = Arc::new(RwLock::new(canon.clone()));
            self.versions.insert(txn_id, version.clone());
            Ok(version)
        }
    }

    fn try_get(&mut self, txn_id: TxnId) -> TCResult<Arc<RwLock<T>>> {
        if let Some(version) = self.versions.get(&txn_id) {
            Ok(version.clone())
        } else {
            if txn_id < self.last_commit {
                return Err(TCError::conflict(format!(
                    "version {} has already been finalized",
                    txn_id
                )));
            }

            let canon = self
                .canon
                .try_read()
                .map_err(|_| TCError::conflict("canonical version is locked"))?;

            let version = Arc::new(RwLock::new(canon.clone()));
            self.versions.insert(txn_id, version.clone());
            Ok(version)
        }
    }
}

impl<T: PartialEq + Clone> Versions<T> {
    async fn commit(&mut self, txn_id: TxnId) -> OwnedRwLockReadGuard<T> {
        let mut canon = self.canon.clone().write_owned().await;
        if let Some(version) = self.versions.get(&txn_id) {
            let version = version.read().await;
            if &*version != &*canon {
                *canon = version.clone();
                self.last_commit = txn_id;
            }
        }

        canon.downgrade()
    }

    fn finalize(&mut self, txn_id: &TxnId) {
        self.versions.remove(txn_id);
    }
}

pub struct TxnLockReadGuard<T: PartialEq + Clone> {
    lock: TxnLock<T>,
    txn_id: TxnId,
    version: Arc<RwLock<T>>,
    guard: OwnedRwLockReadGuard<T>,
}

impl<T: PartialEq + Clone> TxnLockReadGuard<T> {
    #[inline]
    pub fn id(&self) -> &TxnId {
        &self.txn_id
    }
}

impl<T> Clone for TxnLockReadGuard<T>
where
    T: PartialEq + Clone,
{
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

        let guard = self
            .version
            .clone()
            .try_read_owned()
            .expect("TxnLockReadGuard::clone");

        Self {
            lock: self.lock.clone(),
            txn_id: self.txn_id,
            version: self.version.clone(),
            guard,
        }
    }
}

impl<T: PartialEq + Clone> Deref for TxnLockReadGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guard.deref()
    }
}

impl<T: PartialEq + Clone> Drop for TxnLockReadGuard<T> {
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

pub struct TxnLockReadGuardExclusive<T: PartialEq + Clone> {
    lock: TxnLock<T>,
    txn_id: TxnId,
    guard: OwnedRwLockWriteGuard<T>,
    pending_upgrade: bool,
}

impl<T: PartialEq + Clone> TxnLockReadGuardExclusive<T> {
    #[inline]
    pub fn id(&self) -> &TxnId {
        &self.txn_id
    }
}

impl<T: PartialEq + Clone> TxnLockReadGuardExclusive<T> {
    pub fn upgrade(mut self) -> TxnLockWriteGuard<T> {
        let lock = self.lock.clone();
        let txn_id = self.txn_id;

        self.pending_upgrade = true;
        std::mem::drop(self);

        lock.try_write(txn_id).expect("upgrade exclusive read lock")
    }
}

impl<T: PartialEq + Clone> Deref for TxnLockReadGuardExclusive<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl<T> Drop for TxnLockReadGuardExclusive<T>
where
    T: PartialEq + Clone,
{
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

pub struct TxnLockWriteGuard<T: PartialEq + Clone> {
    lock: TxnLock<T>,
    txn_id: TxnId,
    guard: OwnedRwLockWriteGuard<T>,
    pending_downgrade: bool,
}

impl<T: PartialEq + Clone> TxnLockWriteGuard<T> {
    #[inline]
    pub fn id(&self) -> &TxnId {
        &self.txn_id
    }
}

impl<T: PartialEq + Clone> TxnLockWriteGuard<T> {
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

impl<T: PartialEq + Clone> Deref for TxnLockWriteGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guard.deref()
    }
}

impl<T: PartialEq + Clone> DerefMut for TxnLockWriteGuard<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard.deref_mut()
    }
}

impl<T: PartialEq + Clone> Drop for TxnLockWriteGuard<T> {
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

pub struct TxnLockCommitGuard<T> {
    guard: OwnedRwLockReadGuard<T>,
    txn_id: TxnId,
}

impl<T> TxnLockCommitGuard<T> {
    #[inline]
    pub fn id(&self) -> &TxnId {
        &self.txn_id
    }
}

impl<T> Deref for TxnLockCommitGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guard.deref()
    }
}

struct LockState {
    readers: BTreeMap<TxnId, usize>,
    exclusive: BTreeSet<TxnId>,
    writer: Option<TxnId>,
    pending_writes: BTreeSet<TxnId>,
    last_commit: TxnId,
}

impl LockState {
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

    fn can_read(&mut self, txn_id: &TxnId, exclusive: bool) -> bool {
        if self.exclusive.contains(txn_id) {
            debug!("TxnLock is locked exclusively for reading");
            return false;
        }

        for reserved in self.pending_writes.iter().rev() {
            debug_assert!(reserved <= &txn_id);

            if reserved > &self.last_commit && reserved < &txn_id {
                // if there's a pending write that can change the value at this txn_id, wait it out
                info!("TxnLock at {} waiting on a pending write at {}", txn_id, reserved);
                return false;
            }
        }

        if self.writer.as_ref() == Some(txn_id) {
            // if there's an active writer for this txn_id, wait it out
            debug!("TxnLock waiting on a write lock at {}", txn_id);
            return false;
        }

        let num_readers = self.readers.entry(*txn_id).or_insert(0);

        if exclusive {
            if *num_readers == 0 {
                self.exclusive.insert(*txn_id);
            } else {
                trace!("TxnLock is locked non-exclusively for reading");
                return false;
            }
        }

        *num_readers += 1;

        true
    }

    fn try_write(&mut self, txn_id: &TxnId) -> TCResult<bool> {
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
                    "can't acquire write lock at {} since there is already a read lock at {}",
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

                return Ok(false);
            }
        }

        if let Some(writer) = &self.writer {
            assert!(writer <= txn_id);
            // if there's an active write lock, wait it out
            debug!("TxnLock has an active write lock at {}", writer);
            return Ok(false);
        } else if let Some(readers) = self.readers.get(&txn_id) {
            // if there's an active read lock for this txn_id, wait it out
            if readers > &0 {
                debug!("TxnLock has {} active readers at {}", readers, txn_id);
                return Ok(false);
            }
        }

        self.writer = Some(*txn_id);
        self.pending_writes.insert(*txn_id);

        Ok(true)
    }
}

impl LockState {
    fn commit(&mut self, txn_id: TxnId) {
        self.pending_writes.remove(&txn_id);
        self.last_commit = txn_id;
    }

    fn finalize(&mut self, txn_id: &TxnId) {
        self.pending_writes.remove(txn_id);
    }
}

struct Inner<T> {
    name: String,
    state: std::sync::Mutex<LockState>,
    versions: Mutex<Versions<T>>,
    notify: Arc<Notify>,
}

#[derive(Clone)]
pub struct TxnLock<T> {
    inner: Arc<Inner<T>>,
}

impl<T> TxnLock<T> {
    /// Create a new transactional lock.
    pub fn new<I: fmt::Display>(name: I, canon: T) -> Self {
        let state = LockState {
            readers: BTreeMap::new(),
            exclusive: BTreeSet::new(),
            writer: None,
            pending_writes: BTreeSet::new(),
            last_commit: crate::id::MIN_ID,
        };

        Self {
            inner: Arc::new(Inner {
                name: name.to_string(),
                state: std::sync::Mutex::new(state),
                versions: Mutex::new(Versions::new(canon)),
                notify: Arc::new(Notify::new()),
            }),
        }
    }

    /// Get the [`TxnId`] of the last commit to this [`TxnLock`].
    pub fn last_commit(&self) -> TxnId {
        let state = self.inner.state.lock().expect("TxnLock state");
        state.last_commit
    }

    fn wake(&self) {
        self.inner.notify.notify_waiters()
    }
}

impl<T: PartialEq + Clone> TxnLock<T> {
    /// Lock this value for reading at the given `txn_id`.
    pub async fn read(&self, txn_id: TxnId) -> TCResult<TxnLockReadGuard<T>> {
        debug!("lock {} to read at {}...", self.inner.name, txn_id);

        let version = {
            let mut versions = loop {
                {
                    let versions = self.inner.versions.lock().await;
                    let mut lock_state = self.inner.state.lock().expect("TxnLock state to read");
                    if lock_state.can_read(&txn_id, false) {
                        break versions;
                    }
                };

                self.inner.notify.notified().await;
            };

            versions.get(txn_id).await?
        };

        let guard = version.clone().read_owned().await;
        let guard = TxnLockReadGuard {
            lock: self.clone(),
            txn_id,
            version,
            guard,
        };

        debug!("locked {} for reading at {}", self.inner.name, txn_id);
        Ok(guard)
    }

    /// Lock this value for reading at the given `txn_id`, if possible.
    pub fn try_read(&self, txn_id: TxnId) -> TCResult<TxnLockReadGuard<T>> {
        debug!("try to lock {} to read at {}...", self.inner.name, txn_id);

        const ERR: &str = "could not acquire transactional read lock";

        let version = {
            let mut versions = self
                .inner
                .versions
                .try_lock()
                .map_err(|cause| TCError::conflict(format!("{}: {}", ERR, cause)))?;

            let mut lock_state = self.inner.state.lock().expect("TxnLock state to read");
            if lock_state.can_read(&txn_id, false) {
                versions.try_get(txn_id)
            } else {
                Err(TCError::conflict(ERR))
            }
        }?;

        let guard = version
            .clone()
            .try_read_owned()
            .map_err(|cause| TCError::conflict(format!("{}: {}", ERR, cause)))?;

        let guard = TxnLockReadGuard {
            lock: self.clone(),
            txn_id,
            version,
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
            let mut versions = loop {
                {
                    let versions = self.inner.versions.lock().await;
                    let mut lock_state = self
                        .inner
                        .state
                        .lock()
                        .expect("TxnLock state to read exclusively");

                    if lock_state.can_read(&txn_id, true) {
                        break versions;
                    }
                }

                self.inner.notify.notified().await;
            };

            versions.get(txn_id).await?
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
            let mut versions = self
                .inner
                .versions
                .try_lock()
                .map_err(|cause| TCError::conflict(format!("{}: {}", ERR, cause)))?;

            let mut lock_state = self.inner.state.lock().expect("TxnLock state to write");
            if lock_state.can_read(&txn_id, true) {
                versions.try_get(txn_id)
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

    /// Lock this value for writing at the given `txn_id`.
    pub async fn write(&self, txn_id: TxnId) -> TCResult<TxnLockWriteGuard<T>> {
        debug!("locking {} for writing at {}...", self.inner.name, txn_id);

        let version = {
            let mut versions = loop {
                {
                    let versions = self.inner.versions.lock().await;
                    let mut lock_state = self.inner.state.lock().expect("TxnLock::await_writable");
                    if lock_state.try_write(&txn_id)? {
                        break versions;
                    };
                }

                self.inner.notify.notified().await;
            };

            versions.get(txn_id).await?
        };

        let guard = version.write_owned().await;
        let guard = TxnLockWriteGuard::new(self.clone(), txn_id, guard);

        debug!("locked {} for writing at {}", self.inner.name, txn_id);
        Ok(guard)
    }

    /// Try to acquire a write lock synchronously, if possible.
    pub fn try_write(&self, txn_id: TxnId) -> TCResult<TxnLockWriteGuard<T>> {
        const ERR: &str = "could not acquire transactional read lock";

        let version = {
            let mut versions = self
                .inner
                .versions
                .try_lock()
                .map_err(|cause| TCError::conflict(format!("{}: {}", ERR, cause)))?;

            let mut lock_state = self.inner.state.lock().expect("TxnLock::await_readable");
            if lock_state.try_write(&txn_id)? {
                versions.try_get(txn_id)
            } else {
                Err(TCError::conflict(ERR))
            }
        }?;

        let guard = version
            .try_write_owned()
            .map_err(|cause| TCError::conflict(format!("{}: {}", ERR, cause)))?;

        Ok(TxnLockWriteGuard::new(self.clone(), txn_id, guard))
    }
}

#[async_trait]
impl<T: PartialEq + Clone + Send + Sync> Transact for TxnLock<T> {
    type Commit = TxnLockCommitGuard<T>;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        debug!("TxnLock::commit {} at {}", self.inner.name, txn_id);

        let guard = {
            let mut versions = self.inner.versions.lock().await;
            versions.commit(*txn_id).await
        };

        {
            let mut lock_state = self.inner.state.lock().expect("TxnLock::commit");
            lock_state.commit(*txn_id);
        };

        self.wake();

        TxnLockCommitGuard {
            guard,
            txn_id: *txn_id,
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("finalize {} at {}", self.inner.name, txn_id);

        let mut versions = self.inner.versions.lock().await;
        let mut lock_state = self.inner.state.lock().expect("TxnLock::finalize");
        lock_state.finalize(txn_id);
        versions.finalize(txn_id);

        self.wake();
    }
}
