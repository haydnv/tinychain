//! A [`TxnMapLock`] to support transaction-specific versioning of a collection of states.

use std::borrow::Borrow;
use std::collections::hash_map::{Entry, HashMap};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{Arc, Mutex};

use log::{debug, trace, warn};
use tokio::sync::broadcast::{self, Sender};
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use tc_error::*;
use tcgeneric::{Id, Map};

use crate::{TxnId, MIN_ID};

use super::{Versions, Wake};

pub struct TxnMapLockReadGuard<T> {
    lock: TxnMapLock<T>,
    txn_id: TxnId,
    guard: OwnedRwLockReadGuard<BTreeSet<Id>>,
}

impl<T: Clone> TxnMapLockReadGuard<T> {
    pub fn get<K: Borrow<Id>>(&self, key: &Id) -> Option<T> {
        let mut lock_state = self.lock.inner.state.lock().expect("TxnMapLock state");
        let version = lock_state.values.get_mut(key)?;

        version.read(self.txn_id);
        Some(version.get(&self.txn_id))
    }
}

impl<T> Drop for TxnMapLockReadGuard<T> {
    fn drop(&mut self) {
        trace!("TxnLockReadGuard::drop {}", self.lock.inner.name);

        let num_readers = {
            let mut lock_state = self
                .lock
                .inner
                .state
                .lock()
                .expect("TxnLockReadGuard::drop");

            lock_state.drop_read(&self.txn_id)
        };

        if num_readers == 0 {
            self.lock.wake();
        }
    }
}

pub struct TxnMapLockWriteGuard<T> {
    lock: TxnMapLock<T>,
    txn_id: TxnId,
    guard: OwnedRwLockWriteGuard<BTreeSet<Id>>,
}

impl<T: Clone> TxnMapLockWriteGuard<T> {
    pub fn get<K: Borrow<Id>>(&self, key: &Id) -> Option<T> {
        if !self.guard.contains(key) {
            return None;
        }

        let mut lock_state = self.lock.inner.state.lock().expect("TxnMapLock state");
        let version = lock_state.values.get_mut(key)?;

        version.read(self.txn_id);
        Some(version.get(&self.txn_id))
    }

    pub fn insert(&mut self, key: Id, value: T) -> bool {
        let mut lock_state = self.lock.inner.state.lock().expect("TxnMapLock state");

        if self.guard.insert(key.clone()) {
            lock_state
                .values
                .insert(key, Value::version(self.txn_id, value));
            false
        } else {
            let version = lock_state.values.get_mut(&key).expect("value version");
            version.write(self.txn_id, value);
            true
        }
    }

    pub fn remove(&mut self, key: &Id) -> bool {
        self.guard.remove(key)
    }
}

impl<T> Drop for TxnMapLockWriteGuard<T> {
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

        self.lock.wake();
    }
}

struct Value<T> {
    canon: Option<T>,
    versions: HashMap<TxnId, T>,
}

impl<T> Value<T> {
    fn canon(canon: T) -> Self {
        Self {
            canon: Some(canon),
            versions: HashMap::new(),
        }
    }

    fn version(txn_id: TxnId, version: T) -> Self {
        let mut versions = HashMap::new();
        versions.insert(txn_id, version);

        Self {
            canon: None,
            versions,
        }
    }
}

impl<T: Clone> Value<T> {
    fn get(&self, txn_id: &TxnId) -> T {
        self.versions.get(txn_id).expect("value version").clone()
    }

    fn read(&mut self, txn_id: TxnId) {
        if !self.versions.contains_key(&txn_id) {
            let value = self.canon.clone().expect("canonical value");
            self.versions.insert(txn_id, value);
        }
    }

    fn write(&mut self, txn_id: TxnId, value: T) {
        match self.versions.entry(txn_id) {
            Entry::Occupied(mut entry) => {
                *entry.get_mut() = value;
            }
            Entry::Vacant(entry) => {
                entry.insert(value);
            }
        }
    }
}

struct LockState<T> {
    keys: Versions<BTreeSet<Id>>,
    values: BTreeMap<Id, Value<T>>,
    readers: BTreeMap<TxnId, usize>,
    writer: Option<TxnId>,
    pending_writes: BTreeSet<TxnId>,
    last_commit: TxnId,
}

impl<T> LockState<T> {
    fn new(keys: BTreeSet<Id>, values: BTreeMap<Id, Value<T>>) -> Self {
        Self {
            keys: Versions::new(keys),
            values,
            readers: BTreeMap::new(),
            writer: None,
            pending_writes: BTreeSet::new(),
            last_commit: MIN_ID,
        }
    }

    fn drop_read(&mut self, txn_id: &TxnId) -> usize {
        if let Some(writer) = &self.writer {
            assert_ne!(writer, txn_id);
        }

        let num_readers = self.readers.get_mut(txn_id).expect("read lock count");
        *num_readers -= 1;

        trace!(
            "txn lock has {} readers remaining after dropping one read guard",
            num_readers
        );

        *num_readers
    }

    fn try_read(&mut self, txn_id: &TxnId) -> TCResult<Option<Arc<RwLock<BTreeSet<Id>>>>> {
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

        if !self.keys.versions.contains_key(&txn_id) {
            if txn_id <= &self.last_commit {
                // if the requested time is too old, just return an error
                return Err(TCError::conflict(format!(
                    "transaction {} is already finalized, can't acquire read lock",
                    txn_id
                )));
            }
        }

        let num_readers = self.readers.entry(*txn_id).or_insert(0);
        *num_readers += 1;

        Ok(Some(self.keys.get(*txn_id).clone()))
    }

    fn try_write(&mut self, txn_id: &TxnId) -> TCResult<Option<Arc<RwLock<BTreeSet<Id>>>>> {
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

        Ok(Some(self.keys.get(*txn_id).clone()))
    }
}

struct Inner<T> {
    name: String,
    state: Mutex<LockState<T>>,
    tx: Sender<Wake>,
}

#[derive(Clone)]
struct TxnMapLock<T> {
    inner: Arc<Inner<T>>,
}

impl<T> TxnMapLock<T> {
    pub fn new<I: fmt::Display>(name: I) -> Self {
        let (tx, _) = broadcast::channel(16);

        Self {
            inner: Arc::new(Inner {
                name: name.to_string(),
                state: Mutex::new(LockState::new(BTreeSet::new(), BTreeMap::new())),
                tx,
            }),
        }
    }

    pub fn with_contents<I: fmt::Display>(name: I, contents: Map<T>) -> Self {
        let (tx, _) = broadcast::channel(16);

        let mut keys = BTreeSet::new();
        let mut values = BTreeMap::new();
        for (k, v) in contents.into_iter() {
            keys.insert(k.clone());
            values.insert(k, Value::canon(v));
        }

        Self {
            inner: Arc::new(Inner {
                name: name.to_string(),
                state: Mutex::new(LockState::new(keys, values)),
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

impl<T: Clone> TxnMapLock<T> {
    /// Lock this map for reading at the given `txn_id`.
    pub async fn read(&self, txn_id: TxnId) -> TCResult<TxnMapLockReadGuard<T>> {
        debug!("lock map {} to read at {}...", self.inner.name, txn_id);

        let version = {
            let mut rx = self.inner.tx.subscribe();
            loop {
                {
                    let mut lock_state = self.inner.state.lock().expect("TxnMapLock state to read");
                    if let Some(version) = lock_state.try_read(&txn_id)? {
                        break version;
                    }
                }

                if let Err(cause) = rx.recv().await {
                    warn!("TxnMapLock wake error: {}", cause);
                }
            }
        };

        let guard = version.read_owned().await;
        let guard = TxnMapLockReadGuard {
            lock: self.clone(),
            txn_id,
            guard,
        };

        debug!("locked map {} for reading at {}", self.inner.name, txn_id);
        Ok(guard)
    }

    /// Lock this map for writing at the given `txn_id`.
    pub async fn write(&self, txn_id: TxnId) -> TCResult<TxnMapLockWriteGuard<T>> {
        debug!(
            "locking map {} for writing at {}...",
            self.inner.name, txn_id
        );

        let version = {
            let mut rx = self.inner.tx.subscribe();
            loop {
                {
                    let mut lock_state =
                        self.inner.state.lock().expect("TxnMapLock state to write");
                    if let Some(version) = lock_state.try_write(&txn_id)? {
                        break version;
                    };
                }

                if let Err(cause) = rx.recv().await {
                    warn!("TxnMapLock wake error: {}", cause);
                }
            }
        };

        let guard = version.write_owned().await;
        let guard = TxnMapLockWriteGuard {
            lock: self.clone(),
            txn_id,
            guard,
        };

        debug!("locked {} for writing at {}", self.inner.name, txn_id);
        Ok(guard)
    }
}
