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

use crate::class::TCResult;
use crate::error;

use super::{Transact, TxnId};

#[async_trait]
pub trait Mutate: Send {
    type Pending: Clone + Send;

    fn diverge(&self, txn_id: &TxnId) -> Self::Pending;

    async fn converge(&mut self, new_value: Self::Pending);
}

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

pub struct TxnLockReadGuard<T: Mutate> {
    txn_id: TxnId,
    lock: TxnLock<T>,
}

impl<T: Mutate> TxnLockReadGuard<T> {
    pub fn txn_id(&'_ self) -> &'_ TxnId {
        &self.txn_id
    }

    pub fn upgrade(self) -> TxnLockWriteFuture<T> {
        TxnLockWriteFuture {
            txn_id: self.txn_id.clone(),
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
                .lock()
                .unwrap()
                .value_at
                .get(&self.txn_id)
                .unwrap()
                .get()
        }
    }
}

impl<T: Mutate> Drop for TxnLockReadGuard<T> {
    fn drop(&mut self) {
        let lock = &mut self.lock.inner.lock().unwrap();
        match lock.state.readers.get_mut(&self.txn_id) {
            Some(count) if *count > 1 => (*count) -= 1,
            Some(1) => {
                lock.state.readers.remove(&self.txn_id);

                while let Some(waker) = lock.state.wakers.pop_front() {
                    waker.wake()
                }

                lock.state.wakers.shrink_to_fit()
            }
            _ => panic!("TxnLockReadGuard count updated incorrectly!"),
        }
    }
}

pub struct TxnLockWriteGuard<T: Mutate> {
    txn_id: TxnId,
    lock: TxnLock<T>,
}

impl<T: Mutate> TxnLockWriteGuard<T> {
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
                .lock()
                .unwrap()
                .value_at
                .get(&self.txn_id)
                .unwrap()
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
                .lock()
                .unwrap()
                .value_at
                .get_mut(&self.txn_id)
                .unwrap()
                .get()
        }
    }
}

impl<T: Mutate> Drop for TxnLockWriteGuard<T> {
    fn drop(&mut self) {
        let lock = &mut self.lock.inner.lock().unwrap();
        lock.state.reserved = None;

        while let Some(waker) = lock.state.wakers.pop_front() {
            waker.wake()
        }

        lock.state.wakers.shrink_to_fit();
    }
}

struct LockState {
    last_commit: Option<TxnId>,
    readers: BTreeMap<TxnId, usize>,
    reserved: Option<TxnId>,
    wakers: VecDeque<Waker>,
}

struct Inner<T: Mutate> {
    state: LockState,
    value: UnsafeCell<T>,
    value_at: BTreeMap<TxnId, UnsafeCell<<T as Mutate>::Pending>>,
}

pub struct TxnLock<T: Mutate> {
    name: String,
    inner: Arc<Mutex<Inner<T>>>,
}

impl<T: Mutate> Clone for TxnLock<T> {
    fn clone(&self) -> Self {
        TxnLock {
            name: self.name.clone(),
            inner: self.inner.clone(),
        }
    }
}

impl<T: Mutate> TxnLock<T> {
    pub fn new<I: fmt::Display>(name: I, value: T) -> TxnLock<T> {
        let state = LockState {
            last_commit: None,
            readers: BTreeMap::new(),
            reserved: None,
            wakers: VecDeque::new(),
        };

        let inner = Inner {
            state,
            value: UnsafeCell::new(value),
            value_at: BTreeMap::new(),
        };

        TxnLock {
            name: name.to_string(),
            inner: Arc::new(Mutex::new(inner)),
        }
    }

    pub fn canonical(&'_ self) -> &'_ T {
        unsafe { self.inner.lock().unwrap().value.get().as_ref().unwrap() }
    }

    pub fn try_read(&self, txn_id: &TxnId) -> TCResult<Option<TxnLockReadGuard<T>>> {
        debug!("TxnLock::try_read {} at {}", &self.name, txn_id);

        let lock = &mut self.inner.lock().unwrap();

        if !lock.value_at.contains_key(txn_id)
            && txn_id < lock.state.last_commit.as_ref().unwrap_or(&TxnId::zero())
        {
            // If the requested time is too old, just return an error.
            // We can't keep track of every historical version here.
            Err(error::conflict())
        } else if lock.state.reserved.is_some() && txn_id >= lock.state.reserved.as_ref().unwrap() {
            debug!(
                "TxnLock {} is already reserved for writing at {}",
                &self.name,
                lock.state.reserved.as_ref().unwrap()
            );
            // If a writer can mutate the locked value at the requested time, wait it out.
            Ok(None)
        } else {
            // Otherwise, return a ReadGuard.
            if !lock.value_at.contains_key(txn_id) {
                debug!(
                    "setting lock value for TxnLock {} at {}",
                    &self.name, txn_id
                );
                let value_at_txn_id =
                    UnsafeCell::new(unsafe { (&*lock.value.get()).diverge(txn_id) });
                lock.value_at.insert(txn_id.clone(), value_at_txn_id);
            }

            *lock.state.readers.entry(txn_id.clone()).or_insert(0) += 1;
            Ok(Some(TxnLockReadGuard {
                txn_id: txn_id.clone(),
                lock: self.clone(),
            }))
        }
    }

    pub fn read<'a>(&self, txn_id: &'a TxnId) -> TxnLockReadFuture<'a, T> {
        TxnLockReadFuture {
            txn_id,
            lock: self.clone(),
        }
    }

    pub fn try_write<'a>(&self, txn_id: &'a TxnId) -> TCResult<Option<TxnLockWriteGuard<T>>> {
        debug!("TxnLock::try_write {} at {}", &self.name, txn_id);

        let lock = &mut self.inner.lock().unwrap();
        let latest_reader = lock.state.readers.keys().max();

        if latest_reader.is_some() && latest_reader.unwrap() > txn_id {
            // If there's already a reader in the future, there's no point in waiting.
            return Err(error::conflict());
        }

        match &lock.state.reserved {
            // If there's already a writer in the future, there's no point in waiting.
            Some(current_txn) if current_txn > txn_id => Err(error::conflict()),
            // If there's a writer in the past, wait for it to complete.
            Some(current_txn) if current_txn < txn_id => {
                debug!(
                    "TxnLock::write {} at {} blocked on {}",
                    &self.name, txn_id, current_txn
                );
                Ok(None)
            }
            // If there's already a writer for the current transaction, wait for it to complete.
            Some(_) => {
                debug!(
                    "TxnLock::write {} at {} waiting on existing write lock",
                    &self.name, txn_id,
                );
                Ok(None)
            }
            _ => {
                debug!("reserving write lock for {} at {}", &self.name, txn_id);
                // Otherwise, copy the value to be mutated in this transaction.
                lock.state.reserved = Some(txn_id.clone());
                if !lock.value_at.contains_key(txn_id) {
                    let mutation = UnsafeCell::new(unsafe { (&*lock.value.get()).diverge(txn_id) });
                    lock.value_at.insert(txn_id.clone(), mutation);
                }

                Ok(Some(TxnLockWriteGuard {
                    txn_id: txn_id.clone(),
                    lock: self.clone(),
                }))
            }
        }
    }

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
        debug!("TxnLock::commit {} at {}", &self.name, txn_id);

        async {
            debug!(
                "TxnLock::commit {} getting read lock at {}...",
                &self.name, txn_id
            );
            self.read(txn_id).await.unwrap(); // make sure there's no active write lock
            let lock = &mut self.inner.lock().unwrap();
            assert!(lock.state.reserved.is_none());
            if let Some(last_commit) = &lock.state.last_commit {
                assert!(last_commit < txn_id);
            }

            debug!("got inner lock for {}: {}", &self.name, txn_id);
            lock.state.last_commit = Some(txn_id.clone());
            debug!("freed write lock reservation {} at {}", &self.name, txn_id);

            debug!("updating value of {}", &self.name);
            let new_value = if let Some(cell) = lock.value_at.get(txn_id) {
                let pending: &<T as Mutate>::Pending = unsafe { &*cell.get() };
                Some(pending.clone())
            } else {
                None
            };

            if let Some(new_value) = new_value {
                let value = unsafe { &mut *lock.value.get() };
                value.converge(new_value)
            } else {
                Box::pin(future::ready(()))
            }
        }
        .await
        .await;

        let lock = &mut self.inner.lock().unwrap();

        while let Some(waker) = lock.state.wakers.pop_front() {
            waker.wake()
        }

        lock.state.wakers.shrink_to_fit()
    }

    async fn rollback(&self, _txn_id: &TxnId) {
        // no-op
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("TxnLock::finalize {}: {}", &self.name, txn_id);
        self.read(txn_id).await.unwrap(); // make sure there's no active write lock
        let lock = &mut self.inner.lock().unwrap();
        lock.value_at.remove(txn_id);
    }
}

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
                    .lock()
                    .unwrap()
                    .state
                    .wakers
                    .push_back(context.waker().clone());

                Poll::Pending
            }
        }
    }
}

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
                    .lock()
                    .unwrap()
                    .state
                    .wakers
                    .push_back(context.waker().clone());

                Poll::Pending
            }
        }
    }
}
