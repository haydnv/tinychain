//! A [`TxnLock`] to support transaction-specific versioning

use std::cell::UnsafeCell;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};

use async_trait::async_trait;
use futures::Future;
use log::{debug, info};

use tc_error::*;

use super::{Transact, TxnId};

/// An immutable read guard for a transactional state.
pub struct TxnLockReadGuard<T> {
    lock: TxnLock<T>,
    txn_id: TxnId,
}

impl<T> TxnLockReadGuard<T> {
    fn new(lock: TxnLock<T>, txn_id: TxnId) -> Self {
        Self { lock, txn_id }
    }
}

impl<T> Deref for TxnLockReadGuard<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe {
            let state = self.lock.lock_inner("TxnLockReadGuard::deref");
            &*state
                .versions
                .get(&self.txn_id)
                .expect("TxnLock read version")
                .get()
        }
    }
}

impl<T> Drop for TxnLockReadGuard<T> {
    fn drop(&mut self) {
        debug!("TxnLockReadGuard::drop {}", self.lock.inner.name);

        let mut state = self.lock.lock_inner("TxnLockReadGuard::drop");
        debug!("TxnLockReadGuard::drop {} locked state", self.lock.inner.name);

        assert_ne!(state.writer, Some(self.txn_id));

        let num_readers = state
            .readers
            .get_mut(&self.txn_id)
            .expect("read lock count");

        *num_readers -= 1;

        debug!("TxnLock has {} waiting readers", num_readers);
        if num_readers == &0 {
            state.wake();
        }
    }
}

/// An exclusive write lock for a transactional state.
pub struct TxnLockWriteGuard<T> {
    lock: TxnLock<T>,
    txn_id: TxnId,
}

impl<T> TxnLockWriteGuard<T> {
    fn new(lock: TxnLock<T>, txn_id: TxnId) -> Self {
        Self { lock, txn_id }
    }
}

impl<T: Clone> Deref for TxnLockWriteGuard<T> {
    type Target = T;

    fn deref(&self) -> &T {
        let state = self.lock.lock_inner("TxnLockWriteGuard::deref");

        unsafe {
            &*state
                .versions
                .get(&self.txn_id)
                .expect("TxnLock write version")
                .get()
        }
    }
}

impl<T: Clone> DerefMut for TxnLockWriteGuard<T> {
    fn deref_mut(&mut self) -> &mut T {
        let mut state = self.lock.lock_inner("TxnLockWriteGuard::deref");
        state.pending_writes.insert(self.txn_id);

        unsafe {
            &mut *state
                .versions
                .get(&self.txn_id)
                .expect("TxnLock mutable write version")
                .get()
        }
    }
}

impl<T> Drop for TxnLockWriteGuard<T> {
    fn drop(&mut self) {
        debug!("TxnLockWriteGuard::drop {}", self.lock.inner.name);

        let mut state = self.lock.lock_inner("TxnLockWriteGuard::drop");
        if let Some(readers) = state.readers.get(&self.txn_id) {
            assert_eq!(readers, &0);
        }

        state.writer = None;
        state.wake();
    }
}

struct LockState<T> {
    canon: UnsafeCell<T>,
    versions: BTreeMap<TxnId, UnsafeCell<T>>,
    last_commit: TxnId,
    readers: BTreeMap<TxnId, usize>,
    writer: Option<TxnId>,
    pending_writes: BTreeSet<TxnId>,
    wakers: VecDeque<Waker>,
}

impl<T> LockState<T> {
    fn wake(&mut self) {
        debug!("TxnLock waking {} waiting futures", self.wakers.len());

        while let Some(waker) = self.wakers.pop_front() {
            waker.wake();
        }
    }
}

struct Inner<T> {
    name: String,
    state: Mutex<LockState<T>>,
}

/// A lock which provides transaction-specific versions of the locked state.
pub struct TxnLock<T> {
    inner: Arc<Inner<T>>,
}

impl<T> Clone for TxnLock<T> {
    fn clone(&self) -> Self {
        TxnLock {
            inner: self.inner.clone(),
        }
    }
}

impl<T> TxnLock<T> {
    /// Create a new lock.
    pub fn new<I: fmt::Display>(name: I, value: T) -> TxnLock<T> {
        let state = LockState {
            canon: UnsafeCell::new(value),
            versions: BTreeMap::new(),
            last_commit: super::id::MIN_ID,
            readers: BTreeMap::new(),
            writer: None,
            pending_writes: BTreeSet::new(),
            wakers: VecDeque::new(),
        };

        let inner = Inner {
            name: name.to_string(),
            state: Mutex::new(state),
        };

        TxnLock {
            inner: Arc::new(inner),
        }
    }

    #[inline]
    fn lock_inner(&self, expect: &'static str) -> MutexGuard<LockState<T>> {
        debug!("TxnLock::lock_inner {}", expect);

        self.inner.state.lock().expect(expect)
    }
}

impl<T: Clone + Send> TxnLock<T> {
    /// Try to acquire a read lock.
    pub fn read(&self, txn_id: TxnId) -> TxnLockReadFuture<T> {
        debug!("TxnLock::read");

        TxnLockReadFuture {
            lock: self.clone(),
            txn_id,
        }
    }

    fn try_read(&self, txn_id: TxnId) -> TCResult<Option<TxnLockReadGuard<T>>> {
        debug!("TxnLock::try_read {} at {}", self.inner.name, txn_id);

        let mut state = self.lock_inner("TxnLock::try_read");
        for reserved in state.pending_writes.iter().rev() {
            if reserved > &state.last_commit && reserved < &txn_id {
                // if there's a pending write that can change the value at this txn_id, wait it out
                debug!("TxnLock waiting on a pending write at {}", reserved);
                return Ok(None);
            }
        }

        if state.writer == Some(txn_id) {
            // if there's an active writer for this txn_id, wait it out
            debug!("TxnLock waiting on a write lock at {}", txn_id);
            return Ok(None);
        }

        if !state.versions.contains_key(&txn_id) {
            if txn_id <= state.last_commit {
                // if the requested time is too old, just return an error
                info!(
                    "transaction {} is already finalized, can't acquire read lock",
                    txn_id
                );

                return Err(TCError::conflict());
            }

            let version = UnsafeCell::new(unsafe { (&*state.canon.get()).clone() });
            state.versions.insert(txn_id, version);
        }

        debug!("TxnLock locking {} for read at {}", self.inner.name, txn_id);

        *state.readers.entry(txn_id).or_insert(0) += 1;
        Ok(Some(TxnLockReadGuard::new(self.clone(), txn_id)))
    }

    /// Try to acquire a write lock.
    pub fn write(&self, txn_id: TxnId) -> TxnLockWriteFuture<T> {
        debug!("TxnLock::write");

        TxnLockWriteFuture {
            lock: self.clone(),
            txn_id,
        }
    }

    fn try_write(&self, txn_id: TxnId) -> TCResult<Option<TxnLockWriteGuard<T>>> {
        debug!("TxnLock::try_write {} at {}", self.inner.name, txn_id);

        let mut state = self.lock_inner("TxnLock::try_write");

        if state.last_commit >= txn_id {
            // can't write-lock a committed version
            debug!(
                "TxnLock {} has a commit at {}",
                self.inner.name, state.last_commit
            );

            return Err(TCError::conflict());
        }

        for pending in state.pending_writes.iter().rev() {
            if pending > &txn_id {
                // can't write-lock the past
                debug!(
                    "TxnLock {} has pending write in the future",
                    self.inner.name
                );

                return Err(TCError::conflict());
            } else if pending > &state.last_commit && pending < &txn_id {
                // if there's a past write that might still be committed, wait it out
                debug!(
                    "TxnLock {} has a pending write at {}",
                    self.inner.name, pending
                );

                return Ok(None);
            }
        }

        if let Some(reader) = state.readers.keys().max() {
            if reader > &txn_id {
                // can't write-lock the past
                debug!("TxnLock {} has a read lock in the future", self.inner.name);
                return Err(TCError::conflict());
            }
        }

        if let Some(readers) = state.readers.get(&txn_id) {
            // if there's an active read lock for this txn_id, wait it out
            if readers > &0 {
                debug!(
                    "TxnLock {} has {} active readers at {}",
                    self.inner.name, readers, txn_id
                );

                return Ok(None);
            }
        }

        if let Some(writer) = &state.writer {
            // if there's an active write lock, wait it out
            debug!(
                "TxnLock {} has an active write lock at {}",
                self.inner.name, writer
            );

            return Ok(None);
        }

        debug!("TxnLock locking {} for write at {}", self.inner.name, txn_id);

        if !state.versions.contains_key(&txn_id) {
            let version = UnsafeCell::new(unsafe { (&*state.canon.get()).clone() });
            state.versions.insert(txn_id, version);
        }

        state.writer = Some(txn_id);
        Ok(Some(TxnLockWriteGuard::new(self.clone(), txn_id)))
    }
}

#[async_trait]
impl<T: PartialEq + Clone + Send> Transact for TxnLock<T> {
    async fn commit(&self, txn_id: &TxnId) {
        debug!("TxnLock::commit {} at {}", self.inner.name, txn_id);

        let mut state = self.lock_inner("TxnLock::commit");

        let canon = unsafe { &mut *state.canon.get() };
        let (updated, version) = if let Some(cell) = state.versions.get(txn_id) {
            let version = unsafe { &*cell.get() };

            if state.pending_writes.contains(txn_id) {
                (version != canon, version)
            } else {
                assert!(version == canon);
                (false, version)
            }
        } else {
            // it's valid to request a read lock for the first time between commit & finalize
            // so make sure to keep this txn's version around
            let cell = UnsafeCell::new(canon.clone());
            let version = unsafe { &*cell.get() };
            state.versions.insert(*txn_id, cell);
            (false, version)
        };

        for pending in &state.pending_writes {
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
            let has_been_read = if let Some(reserved) = state.readers.keys().last() {
                reserved > txn_id
            } else {
                false
            };

            if has_been_read {
                assert!(txn_id > &state.last_commit);
            }

            *canon = version.clone();
        }

        state.pending_writes.remove(txn_id);

        if txn_id > &state.last_commit {
            state.last_commit = *txn_id;
            state.wake()
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("TxnLock {} finalize {}", &self.inner.name, txn_id);

        let mut state = self.lock_inner("TxnLock::finalize");
        state.versions.remove(txn_id);
        state.pending_writes.remove(txn_id);
    }
}

pub struct TxnLockReadFuture<T> {
    lock: TxnLock<T>,
    txn_id: TxnId,
}

impl<T: Clone + Send> Future for TxnLockReadFuture<T> {
    type Output = TCResult<TxnLockReadGuard<T>>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.lock.try_read(self.txn_id) {
            Ok(Some(guard)) => Poll::Ready(Ok(guard)),
            Err(cause) => Poll::Ready(Err(cause)),
            Ok(None) => {
                let mut state = self.lock.lock_inner("TxnLockWriteFuture::poll");
                state.wakers.push_back(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

pub struct TxnLockWriteFuture<T> {
    lock: TxnLock<T>,
    txn_id: TxnId,
}

impl<T: Clone + Send> Future for TxnLockWriteFuture<T> {
    type Output = TCResult<TxnLockWriteGuard<T>>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.lock.try_write(self.txn_id) {
            Ok(Some(guard)) => Poll::Ready(Ok(guard)),
            Err(cause) => Poll::Ready(Err(cause)),
            Ok(None) => {
                let mut state = self.lock.lock_inner("TxnLockWriteFuture::poll");
                state.wakers.push_back(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}
