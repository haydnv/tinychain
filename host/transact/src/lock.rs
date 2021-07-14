//! A [`TxnLock`] featuring transaction-specific versioning

use std::cell::UnsafeCell;
use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::future::{self, Future};
use futures::task::{Context, Poll, Waker};
use log::debug;

use tc_error::*;

use super::{Transact, TxnId};

/// Define a way to manage transaction-specific versions of a state.
#[async_trait]
pub trait Mutate: Send {
    /// The type of this state when pending a transaction commit.
    type Pending: Clone + Send;

    /// Create a new transaction-specific version of this state.
    fn diverge(&self, txn_id: &TxnId) -> Self::Pending;

    /// Canonicalize a transaction-specific version.
    async fn converge(&mut self, new_value: Self::Pending);
}

/// A generic impl of [`Mutate`], for convenience.
pub struct Mutable<T: Clone + Send> {
    value: T,
}

impl<T: Clone + Send> Mutable<T> {
    pub fn value(&'_ self) -> &'_ T {
        &self.value
    }
}

#[async_trait]
impl<T: Clone + Send> Mutate for Mutable<T> {
    type Pending = T;

    fn diverge(&self, _txn_id: &TxnId) -> Self::Pending {
        self.value.clone()
    }

    async fn converge(&mut self, new_value: Self::Pending) {
        self.value = new_value;
    }
}

impl<T: Clone + Send> From<T> for Mutable<T> {
    fn from(value: T) -> Mutable<T> {
        Mutable { value }
    }
}

/// An immutable read guard for a transactional state.
pub struct TxnLockReadGuard<T: Mutate> {
    txn_id: TxnId,
    lock: TxnLock<T>,
}

impl<T: Mutate> TxnLockReadGuard<T> {
    /// Upgrade this read lock to a write lock.
    pub fn upgrade(self) -> TxnLockWriteFuture<T> {
        TxnLockWriteFuture {
            txn_id: self.txn_id,
            lock: self.lock.clone(),
        }
    }
}

impl<T: Mutate> Deref for TxnLockReadGuard<T> {
    type Target = <T as Mutate>::Pending;

    fn deref(&self) -> &<T as Mutate>::Pending {
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

impl<T: Mutate> Drop for TxnLockReadGuard<T> {
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
pub struct TxnLockWriteGuard<T: Mutate> {
    txn_id: TxnId,
    lock: TxnLock<T>,
}

impl<T: Mutate> TxnLockWriteGuard<T> {
    /// Downgrade this write lock into a read lock;
    pub fn downgrade(self, txn_id: &'_ TxnId) -> TxnLockReadFuture<T> {
        if txn_id != &self.txn_id {
            panic!("Tried to downgrade into a different transaction!");
        }

        self.lock.read(&txn_id)
    }
}

impl<T: Mutate> Deref for TxnLockWriteGuard<T> {
    type Target = <T as Mutate>::Pending;

    fn deref(&self) -> &<T as Mutate>::Pending {
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

impl<T: Mutate> DerefMut for TxnLockWriteGuard<T> {
    fn deref_mut(&mut self) -> &mut <T as Mutate>::Pending {
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

impl<T: Mutate> Drop for TxnLockWriteGuard<T> {
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

struct LockState<T: Mutate> {
    last_commit: Option<TxnId>,
    readers: BTreeMap<TxnId, usize>,
    reserved: Option<TxnId>,
    wakers: VecDeque<Waker>,

    canon: UnsafeCell<T>,
    at: BTreeMap<TxnId, UnsafeCell<<T as Mutate>::Pending>>,
}

struct Inner<T: Mutate> {
    name: String,
    state: Mutex<LockState<T>>,
}

/// A lock which provides transaction-specific versions of the locked state.
pub struct TxnLock<T: Mutate> {
    inner: Arc<Inner<T>>,
}

impl<T: Mutate> Clone for TxnLock<T> {
    fn clone(&self) -> Self {
        TxnLock {
            inner: self.inner.clone(),
        }
    }
}

impl<T: Mutate> TxnLock<T> {
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
                let value_at_txn_id =
                    UnsafeCell::new(unsafe { (&*state.canon.get()).diverge(txn_id) });

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
                    let mutation =
                        UnsafeCell::new(unsafe { (&*state.canon.get()).diverge(txn_id) });

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
impl<T: Mutate> Transact for TxnLock<T> {
    async fn commit(&self, txn_id: &TxnId) {
        debug!("TxnLock::commit {} at {}", self.inner.name, txn_id);

        async {
            self.read(&txn_id).await.expect("TxnLock commit read lock"); // make sure there's no active write lock
            let mut state = self.inner.state.lock().expect("TxnLock commit state");
            assert!(state.reserved.is_none());
            if let Some(ref last_commit) = state.last_commit {
                assert!(last_commit < &txn_id);
            }

            state.last_commit = Some(*txn_id);

            let new_value = if let Some(cell) = state.at.get(&txn_id) {
                let pending: &<T as Mutate>::Pending = unsafe { &*cell.get() };
                Some(pending.clone())
            } else {
                None
            };

            #[allow(clippy::async_yields_async)]
            if let Some(new_value) = new_value {
                let value = unsafe { &mut *state.canon.get() };
                value.converge(new_value)
            } else {
                Box::pin(future::ready(()))
            }
        }
        .await
        .await;

        let mut state = self.inner.state.lock().expect("TxnLock post-commit state");

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
pub struct TxnLockReadFuture<'a, T: Mutate> {
    txn_id: &'a TxnId,
    lock: TxnLock<T>,
}

impl<'a, T: Mutate> Future for TxnLockReadFuture<'a, T> {
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
pub struct TxnLockWriteFuture<T: Mutate> {
    txn_id: TxnId,
    lock: TxnLock<T>,
}

impl<T: Mutate> Future for TxnLockWriteFuture<T> {
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
