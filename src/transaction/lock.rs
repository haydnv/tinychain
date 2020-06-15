use std::cell::UnsafeCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures::future::Future;
use futures::task::{Context, Poll, Waker};

use super::{Transact, TxnId};

pub struct TxnLockReadGuard<T: Clone + Transact> {
    txn_id: TxnId,
    lock: TxnLock<T>,
}

impl<T: Clone + Transact> Deref for TxnLockReadGuard<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.inner.value.get() }
    }
}

impl<T: Clone + Transact> Drop for TxnLockReadGuard<T> {
    fn drop(&mut self) {
        let mut lock_state = self.lock.inner.lock_state.lock().unwrap();
        match lock_state.readers.get_mut(&self.txn_id) {
            Some(count) if *count > 1 => (*count) -= 1,
            Some(1) => {
                lock_state.readers.remove(&self.txn_id);

                while let Some(waker) = lock_state.wakers.pop_front() {
                    waker.wake()
                }
            }
            _ => panic!("TxnLockReadGuard count updated incorrectly!"),
        }
    }
}

pub struct TxnLockWriteGuard<T: Clone + Transact> {
    txn_id: TxnId,
    lock: TxnLock<T>,
}

impl<T: Clone + Transact> Deref for TxnLockWriteGuard<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.inner.value.get() }
    }
}

impl<T: Clone + Transact> DerefMut for TxnLockWriteGuard<T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.inner.value.get() }
    }
}

impl<T: Clone + Transact> Drop for TxnLockWriteGuard<T> {
    fn drop(&mut self) {
        let mut lock_state = self.lock.inner.lock_state.lock().unwrap();
        lock_state.writers.remove(&self.txn_id);

        while let Some(waker) = lock_state.wakers.pop_front() {
            waker.wake()
        }
    }
}

struct LockState {
    readers: HashMap<TxnId, usize>,
    writers: HashSet<TxnId>,
    wakers: VecDeque<Waker>,
}

struct Inner<T: Clone + Transact> {
    value: UnsafeCell<T>,
    lock_state: Mutex<LockState>,
}

#[derive(Clone)]
pub struct TxnLock<T: Clone + Transact> {
    inner: Arc<Inner<T>>,
}

impl<T: Clone + Transact> TxnLock<T> {
    pub fn new(value: T) -> TxnLock<T> {
        let lock_state = LockState {
            readers: HashMap::new(),
            writers: HashSet::new(),
            wakers: VecDeque::new(),
        };

        let inner = Inner {
            value: UnsafeCell::new(value),
            lock_state: Mutex::new(lock_state),
        };

        TxnLock {
            inner: Arc::new(inner),
        }
    }

    pub fn try_read<'a>(&self, txn_id: &'a TxnId) -> Option<TxnLockReadGuard<T>> {
        let mut lock_state = self.inner.lock_state.lock().unwrap();
        if lock_state.writers.contains(&txn_id) {
            None
        } else {
            *lock_state.readers.entry(txn_id.clone()).or_insert(0) += 1;
            Some(TxnLockReadGuard {
                txn_id: txn_id.clone(),
                lock: self.clone(),
            })
        }
    }

    pub fn read<'a>(&self, txn_id: &'a TxnId) -> TxnLockReadFuture<'a, T> {
        TxnLockReadFuture {
            txn_id,
            lock: self.clone(),
        }
    }

    pub fn try_write<'a>(&self, txn_id: &'a TxnId) -> Option<TxnLockWriteGuard<T>> {
        let mut lock_state = self.inner.lock_state.lock().unwrap();
        if lock_state.readers.contains_key(txn_id) || lock_state.writers.contains(txn_id) {
            None
        } else {
            lock_state.writers.insert(txn_id.clone());
            Some(TxnLockWriteGuard {
                txn_id: txn_id.clone(),
                lock: self.clone(),
            })
        }
    }

    pub fn write<'a>(&self, txn_id: &'a TxnId) -> TxnLockWriteFuture<'a, T> {
        TxnLockWriteFuture {
            txn_id,
            lock: self.clone(),
        }
    }
}

impl<T: Clone + Default + Transact> Default for TxnLock<T> {
    fn default() -> TxnLock<T> {
        TxnLock::new(Default::default())
    }
}

pub struct TxnLockReadFuture<'a, T: Clone + Transact> {
    txn_id: &'a TxnId,
    lock: TxnLock<T>,
}

impl<'a, T: Clone + Transact> Future for TxnLockReadFuture<'a, T> {
    type Output = TxnLockReadGuard<T>;

    fn poll(self: Pin<&mut Self>, context: &mut Context) -> Poll<Self::Output> {
        if let Some(guard) = self.lock.try_read(self.txn_id) {
            Poll::Ready(guard)
        } else {
            self.lock
                .inner
                .lock_state
                .lock()
                .unwrap()
                .wakers
                .push_back(context.waker().clone());

            Poll::Pending
        }
    }
}

pub struct TxnLockWriteFuture<'a, T: Clone + Transact> {
    txn_id: &'a TxnId,
    lock: TxnLock<T>,
}

impl<'a, T: Clone + Transact> Future for TxnLockWriteFuture<'a, T> {
    type Output = TxnLockWriteGuard<T>;

    fn poll(self: Pin<&mut Self>, context: &mut Context) -> Poll<Self::Output> {
        if let Some(guard) = self.lock.try_write(self.txn_id) {
            Poll::Ready(guard)
        } else {
            self.lock
                .inner
                .lock_state
                .lock()
                .unwrap()
                .wakers
                .push_back(context.waker().clone());

            Poll::Pending
        }
    }
}
