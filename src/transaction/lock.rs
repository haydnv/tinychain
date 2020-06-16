use std::cell::UnsafeCell;
use std::collections::{BTreeMap, VecDeque};
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures::future::Future;
use futures::task::{Context, Poll, Waker};

use crate::error;
use crate::value::TCResult;

use super::{Txn, TxnId};

pub struct TxnLockReadGuard<T: Clone> {
    txn_id: TxnId,
    lock: TxnLock<T>,
}

impl<T: Clone> Deref for TxnLockReadGuard<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.inner.lock().unwrap().value.get() }
    }
}

impl<T: Clone> Drop for TxnLockReadGuard<T> {
    fn drop(&mut self) {
        let lock = &mut self.lock.inner.lock().unwrap();
        match lock.state.readers.get_mut(&self.txn_id) {
            Some(count) if *count > 1 => (*count) -= 1,
            Some(1) => {
                lock.state.readers.remove(&self.txn_id);

                while let Some(waker) = lock.state.wakers.pop_front() {
                    waker.wake()
                }
            }
            _ => panic!("TxnLockReadGuard count updated incorrectly!"),
        }
    }
}

pub struct TxnLockWriteGuard<T: Clone> {
    lock: TxnLock<T>,
}

impl<T: Clone> Deref for TxnLockWriteGuard<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe {
            &*self
                .lock
                .inner
                .lock()
                .unwrap()
                .mutation
                .as_ref()
                .unwrap()
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
                .lock()
                .unwrap()
                .mutation
                .as_ref()
                .unwrap()
                .get()
        }
    }
}

impl<T: Clone> Drop for TxnLockWriteGuard<T> {
    fn drop(&mut self) {
        let lock = &mut self.lock.inner.lock().unwrap();
        lock.state.writer = None;
        lock.mutation = None;

        while let Some(waker) = lock.state.wakers.pop_front() {
            waker.wake()
        }
    }
}

struct LockState {
    readers: BTreeMap<TxnId, usize>,
    writer: Option<TxnId>,
    wakers: VecDeque<Waker>,
}

struct Inner<T: Clone> {
    state: LockState,
    value: UnsafeCell<T>,
    mutation: Option<UnsafeCell<T>>,
}

#[derive(Clone)]
pub struct TxnLock<T: Clone> {
    inner: Arc<Mutex<Inner<T>>>,
}

impl<T: Clone> TxnLock<T> {
    pub fn new(value: T) -> TxnLock<T> {
        let state = LockState {
            readers: BTreeMap::new(),
            writer: None,
            wakers: VecDeque::new(),
        };

        let inner = Inner {
            state,
            value: UnsafeCell::new(value),
            mutation: None,
        };

        TxnLock {
            inner: Arc::new(Mutex::new(inner)),
        }
    }

    pub fn try_read<'a>(&self, txn_id: &'a TxnId) -> Option<TxnLockReadGuard<T>> {
        let lock = &mut self.inner.lock().unwrap();
        if lock.state.writer.is_some() && lock.state.writer.as_ref().unwrap() <= txn_id {
            None
        } else {
            *lock.state.readers.entry(txn_id.clone()).or_insert(0) += 1;
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

    pub fn try_write<'a>(&self, txn: &'a Txn) -> TCResult<Option<Arc<TxnLockWriteGuard<T>>>> {
        let lock = &mut self.inner.lock().unwrap();

        if (lock.state.writer.is_some() && lock.state.writer.as_ref().unwrap() > txn.id())
            || (!lock.state.readers.is_empty()
                && lock.state.readers.keys().max().unwrap() > txn.id())
        {
            Err(error::conflict())
        } else if lock.state.readers.contains_key(txn.id()) {
            Ok(None)
        } else {
            lock.state.writer = Some(txn.id().clone());
            if lock.mutation.is_none() {
                lock.mutation = Some(UnsafeCell::new(unsafe { (&*lock.value.get()).clone() }));
            }

            let guard = Arc::new(TxnLockWriteGuard { lock: self.clone() });
            Ok(Some(guard))
        }
    }

    pub fn write<'a>(&self, txn: &'a Txn) -> TxnLockWriteFuture<'a, T> {
        TxnLockWriteFuture {
            txn,
            lock: self.clone(),
        }
    }
}

impl<T: Clone + Default> Default for TxnLock<T> {
    fn default() -> TxnLock<T> {
        TxnLock::new(Default::default())
    }
}

pub struct TxnLockReadFuture<'a, T: Clone> {
    txn_id: &'a TxnId,
    lock: TxnLock<T>,
}

impl<'a, T: Clone> Future for TxnLockReadFuture<'a, T> {
    type Output = TxnLockReadGuard<T>;

    fn poll(self: Pin<&mut Self>, context: &mut Context) -> Poll<Self::Output> {
        if let Some(guard) = self.lock.try_read(self.txn_id) {
            Poll::Ready(guard)
        } else {
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

pub struct TxnLockWriteFuture<'a, T: Clone> {
    txn: &'a Txn,
    lock: TxnLock<T>,
}

impl<'a, T: Clone + Send + Sync> Future for TxnLockWriteFuture<'a, T> {
    type Output = TCResult<Arc<TxnLockWriteGuard<T>>>;

    fn poll(self: Pin<&mut Self>, context: &mut Context) -> Poll<Self::Output> {
        match self.lock.try_write(self.txn) {
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
