use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures::future::Future;
use futures::task::{Context, Poll, Waker};

/**
 * This is very similar to futures_locks::RwLock[1] but I couldn't get the version from crates.io to
 * work when I tried it, so I wrote this RwLock based on TxnLock from this crate (which in turn is
 * very loosely based on futures_util::lock::Mutex[2]).
 *
 * [1] https://docs.rs/futures-locks/0.5.0/futures_locks/struct.RwLock.html
 * [2] https://docs.rs/futures-util/0.3.5/futures_util/lock/struct.Mutex.html
 */

pub struct RwLockReadGuard<T> {
    lock: RwLock<T>,
}

impl<T> Deref for RwLockReadGuard<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.inner.lock().unwrap().value.get() }
    }
}

impl<T> Drop for RwLockReadGuard<T> {
    fn drop(&mut self) {
        let lock = &mut self.lock.inner.lock().unwrap();
        lock.state.readers -= 1;

        if lock.state.readers == 0 {
            while let Some(waker) = lock.state.wakers.pop_front() {
                waker.wake()
            }

            lock.state.wakers.shrink_to_fit()
        }
    }
}

pub struct RwLockWriteGuard<T> {
    lock: RwLock<T>,
}

impl<T> Deref for RwLockWriteGuard<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.inner.lock().unwrap().value.get() }
    }
}

impl<T> DerefMut for RwLockWriteGuard<T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.inner.lock().unwrap().value.get() }
    }
}

impl<T> Drop for RwLockWriteGuard<T> {
    fn drop(&mut self) {
        let mut lock = self.lock.inner.lock().unwrap();
        lock.state.writer = false;

        while let Some(waker) = lock.state.wakers.pop_front() {
            waker.wake()
        }

        lock.state.wakers.shrink_to_fit()
    }
}

struct LockState {
    readers: usize,
    writer: bool,
    wakers: VecDeque<Waker>,
}

struct Inner<T> {
    state: LockState,
    value: UnsafeCell<T>,
}

pub struct RwLock<T> {
    inner: Arc<Mutex<Inner<T>>>,
}

impl<T> Clone for RwLock<T> {
    fn clone(&self) -> RwLock<T> {
        RwLock {
            inner: self.inner.clone(),
        }
    }
}

impl<T> RwLock<T> {
    fn new(value: T) -> RwLock<T> {
        let state = LockState {
            readers: 0,
            writer: false,
            wakers: VecDeque::new(),
        };

        let inner = Inner {
            state,
            value: UnsafeCell::new(value),
        };

        RwLock {
            inner: Arc::new(Mutex::new(inner)),
        }
    }

    pub fn try_read(&self) -> Option<RwLockReadGuard<T>> {
        let lock = &mut self.inner.lock().unwrap();
        if lock.state.writer {
            None
        } else {
            lock.state.readers += 1;
            Some(RwLockReadGuard { lock: self.clone() })
        }
    }

    pub fn read(&self) -> RwLockReadFuture<T> {
        RwLockReadFuture { lock: self.clone() }
    }

    pub fn try_write(&self) -> Option<RwLockWriteGuard<T>> {
        let lock = &mut self.inner.lock().unwrap();

        if lock.state.writer || lock.state.readers > 0 {
            None
        } else {
            lock.state.writer = true;
            Some(RwLockWriteGuard { lock: self.clone() })
        }
    }

    pub fn write(&self) -> RwLockWriteFuture<T> {
        RwLockWriteFuture { lock: self.clone() }
    }
}

pub struct RwLockReadFuture<T> {
    lock: RwLock<T>,
}

impl<T> Future for RwLockReadFuture<T> {
    type Output = RwLockReadGuard<T>;

    fn poll(self: Pin<&mut Self>, context: &mut Context) -> Poll<Self::Output> {
        match self.lock.try_read() {
            Some(guard) => Poll::Ready(guard),
            None => {
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

pub struct RwLockWriteFuture<T> {
    lock: RwLock<T>,
}

impl<T> Future for RwLockWriteFuture<T> {
    type Output = RwLockWriteGuard<T>;

    fn poll(self: Pin<&mut Self>, context: &mut Context) -> Poll<Self::Output> {
        match self.lock.try_write() {
            Some(guard) => Poll::Ready(guard),
            None => {
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
