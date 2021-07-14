//! A [`TxnLock`] featuring transaction-specific versioning

use std::cell::UnsafeCell;
use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::future::Future;
use futures::task::{Context, Poll, Waker};
use log::debug;

use tc_error::*;

use super::{Transact, TxnId};

/// An immutable read guard for a transactional state.
pub struct TxnLockReadGuard<T> {
    txn_id: TxnId,
    lock: TxnLock<T>,
}

impl<T> TxnLockReadGuard<T> {
    /// Upgrade this read lock to a write lock.
    pub fn upgrade(self) -> TxnLockWriteFuture<T> {
        TxnLockWriteFuture {
            txn_id: self.txn_id,
            lock: self.lock.clone(),
        }
    }
}

impl<T> Deref for TxnLockReadGuard<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe {
            &*self
                .lock
                .inner
                .state
                .lock()
                .expect("TxnLock read")
                .at
                .get(&self.txn_id)
                .expect("TxnLock read version")
                .get()
        }
    }
}

impl<T> Drop for TxnLockReadGuard<T> {
    fn drop(&mut self) {
        let mut state = self.lock.inner.state.lock().expect("TxnLockReadGuard drop");
        match state.readers.get_mut(&self.txn_id) {
            Some(count) if *count > 1 => (*count) -= 1,
            Some(count) if *count == 1 => {
                *count = 0;

                while let Some(waker) = state.wakers.pop_front() {
                    waker.wake()
                }
            }
            _ => panic!("TxnLockReadGuard count updated incorrectly!"),
        }
    }
}

/// An exclusive write lock for a transactional state.
pub struct TxnLockWriteGuard<T> {
    txn_id: TxnId,
    lock: TxnLock<T>,
}

impl<T: Clone> TxnLockWriteGuard<T> {
    /// Downgrade this write lock into a read lock;
    pub fn downgrade(self, txn_id: &'_ TxnId) -> TxnLockReadFuture<T> {
        if txn_id != &self.txn_id {
            panic!("Tried to downgrade into a different transaction!");
        }

        self.lock.read(&txn_id)
    }
}

impl<T: Clone> Deref for TxnLockWriteGuard<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe {
            &*self
                .lock
                .inner
                .state
                .lock()
                .expect("TxnLock read")
                .at
                .get(&self.txn_id)
                .expect("TxnLock version read")
                .get()
        }
    }
}

impl<T: Clone> DerefMut for TxnLockWriteGuard<T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe {
            &mut *self
                .lock
                .inner
                .state
                .lock()
                .expect("TxnLock write")
                .at
                .get_mut(&self.txn_id)
                .expect("TxnLock version write")
                .get()
        }
    }
}

impl<T> Drop for TxnLockWriteGuard<T> {
    fn drop(&mut self) {
        let mut state = self
            .lock
            .inner
            .state
            .lock()
            .expect("TxnLockWriteGuard drop");

        state.reserved = None;

        while let Some(waker) = state.wakers.pop_front() {
            waker.wake()
        }
    }
}

struct LockState<T> {
    last_commit: Option<TxnId>,
    readers: BTreeMap<TxnId, usize>,
    reserved: Option<TxnId>,
    wakers: VecDeque<Waker>,

    canon: UnsafeCell<T>,
    at: BTreeMap<TxnId, UnsafeCell<T>>,
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

impl<T: Clone> TxnLock<T> {
    /// Create a new lock.
    pub fn new<I: fmt::Display>(name: I, value: T) -> TxnLock<T> {
        let state = LockState {
            last_commit: None,
            readers: BTreeMap::new(),
            reserved: None,
            wakers: VecDeque::new(),

            canon: UnsafeCell::new(value),
            at: BTreeMap::new(),
        };

        let inner = Inner {
            name: name.to_string(),
            state: Mutex::new(state),
        };

        TxnLock {
            inner: Arc::new(inner),
        }
    }

    /// Try to acquire a read lock synchronously.
    pub fn try_read(&self, txn_id: &TxnId) -> TCResult<Option<TxnLockReadGuard<T>>> {
        let mut state = if let Ok(state) = self.inner.state.try_lock() {
            state
        } else {
            return Ok(None);
        };

        let last_commit = state.last_commit.as_ref().unwrap_or(&super::id::MIN_ID);

        if !state.at.contains_key(txn_id) && txn_id < last_commit {
            // If the requested time is too old, just return an error.
            // We can't keep track of every historical version here.
            debug!(
                "transaction {} is already finalized, can't acquire read lock",
                txn_id
            );

            Err(TCError::conflict())
        } else if let Some(ref past_write) = state.reserved {
            // If a writer can mutate the locked value at the requested time, wait it out.
            debug!(
                "TxnLock {} is already reserved for writing at {}",
                &self.inner.name, past_write
            );

            Ok(None)
        } else {
            // Otherwise, return a ReadGuard.
            if !state.at.contains_key(txn_id) {
                let value_at_txn_id = UnsafeCell::new(unsafe { (&*state.canon.get()).clone() });

                state.at.insert(*txn_id, value_at_txn_id);
            }

            *state.readers.entry(*txn_id).or_insert(0) += 1;

            Ok(Some(TxnLockReadGuard {
                txn_id: *txn_id,
                lock: self.clone(),
            }))
        }
    }

    /// Lock this state for reading at the given [`TxnId`].
    pub fn read<'a>(&self, txn_id: &'a TxnId) -> TxnLockReadFuture<'a, T> {
        TxnLockReadFuture {
            txn_id,
            lock: self.clone(),
        }
    }

    /// Try to acquire a write lock, synchronously.
    pub fn try_write(&self, txn_id: &TxnId) -> TCResult<Option<TxnLockWriteGuard<T>>> {
        let mut state = if let Ok(state) = self.inner.state.try_lock() {
            state
        } else {
            return Ok(None);
        };

        if let Some(latest_read) = state.readers.keys().max() {
            // If there's already a reader in the future, there's no point in waiting.
            if latest_read > txn_id {
                return Err(TCError::conflict());
            }
        }

        match &state.reserved {
            // If there's already a writer in the future, there's no point in waiting.
            Some(current_txn) if current_txn > txn_id => Err(TCError::conflict()),
            // If there's a writer in the past, wait for it to complete.
            Some(current_txn) if current_txn < txn_id => {
                debug!(
                    "TxnLock {} at {} blocked on {}",
                    &self.inner.name, txn_id, current_txn
                );

                Ok(None)
            }
            // If there's already a writer for the current transaction, wait for it to complete.
            Some(_) => {
                debug!(
                    "TxnLock {} waiting on existing write lock",
                    &self.inner.name
                );
                Ok(None)
            }
            _ => {
                if let Some(readers) = state.readers.get(txn_id) {
                    if readers > &0 {
                        // There's already a reader for the current transaction, wait for it.
                        return Ok(None);
                    }
                }

                // Otherwise, copy the value to be mutated in this transaction.
                state.reserved = Some(*txn_id);
                if !state.at.contains_key(txn_id) {
                    let mutation = UnsafeCell::new(unsafe { (&*state.canon.get()).clone() });

                    state.at.insert(*txn_id, mutation);
                }

                Ok(Some(TxnLockWriteGuard {
                    txn_id: *txn_id,
                    lock: self.clone(),
                }))
            }
        }
    }

    /// Lock this state for writing at the given [`TxnId`].
    pub fn write(&self, txn_id: TxnId) -> TxnLockWriteFuture<T> {
        TxnLockWriteFuture {
            txn_id,
            lock: self.clone(),
        }
    }
}

#[async_trait]
impl<T: Clone + Send> Transact for TxnLock<T> {
    async fn commit(&self, txn_id: &TxnId) {
        debug!("TxnLock::commit {} at {}", self.inner.name, txn_id);

        let mut state = self.inner.state.lock().expect("TxnLock commit state");
        assert!(state.reserved.is_none());
        if let Some(ref last_commit) = state.last_commit {
            assert!(last_commit < &txn_id);
        }

        state.last_commit = Some(*txn_id);

        if let Some(cell) = state.at.get(txn_id) {
            let value = unsafe { &mut *state.canon.get() };
            let pending: &T = unsafe { &*cell.get() };
            *value = pending.clone();
        }

        while let Some(waker) = state.wakers.pop_front() {
            waker.wake()
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("TxnLock {} finalize {}", &self.inner.name, txn_id);

        let mut state = self.inner.state.lock().expect("TxnLock finalize state");

        if let Some(readers) = state.readers.remove(txn_id) {
            if readers > 0 {
                panic!("tried to finalize a transaction that's still active!")
            }
        }

        state.at.remove(txn_id);
    }
}

/// A read lock future.
pub struct TxnLockReadFuture<'a, T> {
    txn_id: &'a TxnId,
    lock: TxnLock<T>,
}

impl<'a, T: Clone> Future for TxnLockReadFuture<'a, T> {
    type Output = TCResult<TxnLockReadGuard<T>>;

    fn poll(self: Pin<&mut Self>, context: &mut Context) -> Poll<Self::Output> {
        match self.lock.try_read(&self.txn_id) {
            Ok(Some(guard)) => Poll::Ready(Ok(guard)),
            Err(cause) => Poll::Ready(Err(cause)),
            Ok(None) => {
                self.lock
                    .inner
                    .state
                    .lock()
                    .expect("TxnLockReadFuture state")
                    .wakers
                    .push_back(context.waker().clone());

                Poll::Pending
            }
        }
    }
}

/// A write lock future.
pub struct TxnLockWriteFuture<T> {
    txn_id: TxnId,
    lock: TxnLock<T>,
}

impl<T: Clone> Future for TxnLockWriteFuture<T> {
    type Output = TCResult<TxnLockWriteGuard<T>>;

    fn poll(self: Pin<&mut Self>, context: &mut Context) -> Poll<Self::Output> {
        match self.lock.try_write(&self.txn_id) {
            Ok(Some(guard)) => Poll::Ready(Ok(guard)),
            Err(cause) => Poll::Ready(Err(cause)),
            Ok(None) => {
                self.lock
                    .inner
                    .state
                    .lock()
                    .expect("TxnLockWriteFuture state")
                    .wakers
                    .push_back(context.waker().clone());

                Poll::Pending
            }
        }
    }
}
