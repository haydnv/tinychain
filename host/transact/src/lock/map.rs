//! A [`TxnMapLock`] to support transaction-specific versioning of a collection of states.

use std::borrow::Borrow;
use std::collections::hash_map::{Entry, HashMap};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use log::debug;

use tc_error::*;
use tcgeneric::{Id, Map};

use crate::lock::{TxnLock, TxnLockReadGuard, TxnLockWriteGuard};
use crate::{Transact, TxnId};

pub struct TxnMapLockReadGuard<T> {
    lock: TxnMapLock<T>,
    guard: TxnLockReadGuard<BTreeSet<Id>>,
}

impl<T: Clone> TxnMapLockReadGuard<T> {
    pub fn get<K: Borrow<Id>>(&self, key: &Id) -> Option<T> {
        let mut values = self.lock.inner.values.lock().expect("TxnMapLock state");
        let version = values.get_mut(key)?;

        Some(version.read(*self.guard.id()))
    }
}

pub struct TxnMapLockWriteGuard<T> {
    lock: TxnMapLock<T>,
    guard: TxnLockWriteGuard<BTreeSet<Id>>,
}

impl<T: Clone> TxnMapLockWriteGuard<T> {
    pub fn get<K: Borrow<Id>>(&self, key: &Id) -> Option<T> {
        if !self.guard.contains(key) {
            return None;
        }

        let mut values = self.lock.inner.values.lock().expect("TxnMapLock state");
        let version = values.get_mut(key)?;

        Some(version.read(*self.guard.id()))
    }

    pub fn insert(&mut self, key: Id, value: T) -> bool {
        let mut values = self.lock.inner.values.lock().expect("TxnMapLock state");

        if self.guard.insert(key.clone()) {
            values.insert(key, Value::version(*self.guard.id(), value));
            false
        } else {
            let version = values.get_mut(&key).expect("value version");
            version.write(*self.guard.id(), value);
            true
        }
    }

    pub fn remove(&mut self, key: &Id) -> bool {
        self.guard.remove(key)
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
    fn commit(&mut self, txn_id: &TxnId) {
        if let Some(value) = self.versions.get(txn_id) {
            self.canon = Some(value.clone());
        }
    }

    fn finalize(&mut self, txn_id: &TxnId) {
        self.versions.remove(txn_id);
    }

    fn read(&mut self, txn_id: TxnId) -> T {
        if let Some(value) = self.versions.get(&txn_id) {
            value.clone()
        } else {
            let value = self.canon.clone().expect("canonical value");
            self.versions.insert(txn_id, value.clone());
            value
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

struct Inner<T> {
    name: String,
    keys: TxnLock<BTreeSet<Id>>,
    values: Mutex<BTreeMap<Id, Value<T>>>,
}

#[derive(Clone)]
pub struct TxnMapLock<T> {
    inner: Arc<Inner<T>>,
}

impl<T> TxnMapLock<T> {
    pub fn new<I: fmt::Display>(name: I) -> Self {
        Self {
            inner: Arc::new(Inner {
                name: name.to_string(),
                keys: TxnLock::new(format!("{} keys", name), BTreeSet::new()),
                values: Mutex::new(BTreeMap::new()),
            }),
        }
    }

    pub fn with_contents<I: fmt::Display>(name: I, contents: Map<T>) -> Self {
        let mut keys = BTreeSet::new();
        let mut values = BTreeMap::new();
        for (k, v) in contents.into_iter() {
            keys.insert(k.clone());
            values.insert(k, Value::canon(v));
        }

        Self {
            inner: Arc::new(Inner {
                name: name.to_string(),
                keys: TxnLock::new(format!("{} keys", name), BTreeSet::new()),
                values: Mutex::new(values),
            }),
        }
    }
}

impl<T: Clone> TxnMapLock<T> {
    /// Lock this map for reading at the given `txn_id`.
    pub async fn read(&self, txn_id: TxnId) -> TCResult<TxnMapLockReadGuard<T>> {
        debug!("lock map {} to read at {}...", self.inner.name, txn_id);

        let guard = self.inner.keys.read(txn_id).await?;
        let guard = TxnMapLockReadGuard {
            lock: self.clone(),
            guard,
        };

        debug!("locked map {} for reading at {}", self.inner.name, txn_id);
        Ok(guard)
    }

    /// Lock this map for writing at the given `txn_id`.
    pub async fn write(&self, txn_id: TxnId) -> TCResult<TxnMapLockWriteGuard<T>> {
        debug!("lock map {} for writing at {}...", self.inner.name, txn_id);

        let guard = self.inner.keys.write(txn_id).await?;
        let guard = TxnMapLockWriteGuard {
            lock: self.clone(),
            guard,
        };

        debug!("locked {} for writing at {}", self.inner.name, txn_id);
        Ok(guard)
    }
}

#[async_trait]
impl<T: Eq + Clone + Send + Sync> Transact for TxnMapLock<T> {
    async fn commit(&self, txn_id: &TxnId) {
        debug!("commit map {} at {}", self.inner.name, txn_id);

        self.inner.keys.commit_dep(txn_id, |keys| {
            let mut values = self.inner.values.lock().expect("transactional map values");

            for key in keys {
                let value = values.get_mut(key).expect("transactional map value");
                value.commit(txn_id);
            }
        })
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("finalize map {} at {}", self.inner.name, txn_id);

        {
            let keys = self
                .inner
                .keys
                .try_read_exclusive(*txn_id)
                .expect("transactional map keys");

            let mut values = self.inner.values.lock().expect("transactional map values");

            let mut to_delete = Vec::with_capacity(values.len());
            for (key, value) in values.iter_mut() {
                if keys.contains(key) {
                    value.finalize(txn_id);
                } else {
                    if let Some(last_commit) = value.versions.keys().last() {
                        if last_commit <= txn_id {
                            to_delete.push(key.clone());
                        }
                    } else {
                        to_delete.push(key.clone())
                    }
                }
            }

            for key in to_delete.into_iter() {
                values.remove(&key);
            }
        }

        self.inner.keys.finalize(txn_id).await;
    }
}
