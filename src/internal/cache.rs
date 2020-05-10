use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::iter::FromIterator;
use std::sync::RwLock;

use crate::transaction::TransactionId;

pub struct Map<K: Clone + Eq + Hash + Send + Sync, V: Send + Sync> {
    map: RwLock<Single<HashMap<K, V>>>,
}

impl<K: Clone + Eq + Hash + Send + Sync, V: Clone + Send + Sync> Map<K, V> {
    pub fn new() -> Map<K, V> {
        Map {
            map: RwLock::new(Single::new(Some(HashMap::new()))),
        }
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.map.read().unwrap().read().unwrap().contains_key(key)
    }

    pub fn drain(&self) -> HashMap<K, V> {
        self.map
            .write()
            .unwrap()
            .replace(Some(HashMap::new()))
            .unwrap()
    }

    pub fn extend<I: Iterator<Item = (K, V)>>(&self, i: I) {
        let mut lock = self.map.write().unwrap();
        let mut map = lock.replace(None).unwrap();
        map.extend(i);
        lock.replace(Some(map));
    }

    pub fn get(&self, key: &K) -> Option<V> {
        match self.map.read().unwrap().read().unwrap().get(key) {
            Some(val) => Some(val.clone()),
            None => None,
        }
    }

    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let mut lock = self.map.write().unwrap();
        let mut map = lock.replace(None).unwrap();
        let result = map.insert(key, value);
        lock.replace(Some(map));
        result
    }

    pub fn keys(&self) -> HashSet<K> {
        let lock = self.map.read().unwrap();
        let keys = lock.read().unwrap().keys();
        keys.cloned().collect()
    }

    pub fn remove(&self, key: &K) -> Option<V> {
        let mut lock = self.map.write().unwrap();
        let mut map = lock.replace(None).unwrap();
        let result = map.remove(key);
        lock.replace(Some(map));
        result
    }
}

impl<K: Clone + Eq + Hash + Send + Sync, V: Send + Sync> FromIterator<(K, V)> for Map<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(i: T) -> Map<K, V> {
        let mut map: HashMap<K, V> = HashMap::new();
        for (k, v) in i {
            map.insert(k, v);
        }
        Map {
            map: RwLock::new(Single::new(Some(map))),
        }
    }
}

#[derive(Debug)]
pub struct Deque<V> {
    deque: RwLock<VecDeque<V>>,
}

impl<V> Deque<V> {
    pub fn new() -> Deque<V> {
        Deque {
            deque: RwLock::new(VecDeque::new()),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.deque.read().unwrap().is_empty()
    }

    pub fn pop_front(&self) -> Option<V> {
        self.deque.write().unwrap().pop_front()
    }

    pub fn push_back(&self, item: V) {
        self.deque.write().unwrap().push_back(item)
    }
}

pub struct TransactionCache<K: Eq + Hash, V: Clone> {
    cache: RwLock<HashMap<TransactionId, HashMap<K, V>>>,
}

impl<K: Eq + Hash, V: Clone> TransactionCache<K, V> {
    pub fn new() -> TransactionCache<K, V> {
        TransactionCache {
            cache: RwLock::new(HashMap::new()),
        }
    }

    pub fn close(&self, txn_id: &TransactionId) -> HashMap<K, V> {
        println!("TransactionCache closing {}", txn_id);
        self.cache
            .write()
            .unwrap()
            .remove(txn_id)
            .unwrap_or_else(HashMap::new)
    }

    pub fn get(&self, txn_id: &TransactionId, key: &K) -> Option<V> {
        if let Some(entries) = self.cache.read().unwrap().get(txn_id) {
            entries.get(key).cloned()
        } else {
            None
        }
    }

    pub fn insert(&self, txn_id: TransactionId, key: K, value: V) {
        let mut cache = self.cache.write().unwrap();
        if let Some(map) = cache.get_mut(&txn_id) {
            map.insert(key, value);
        } else {
            let mut map: HashMap<K, V> = HashMap::new();
            map.insert(key, value);
            cache.insert(txn_id, map);
        }
    }
}

pub struct Single<T: Send + Sync> {
    item: Vec<T>,
}

impl<T: Send + Sync> Single<T> {
    pub fn new(item: Option<T>) -> Single<T> {
        let mut v = Vec::with_capacity(1);

        if let Some(item) = item {
            v.push(item);
        }

        Single { item: v }
    }

    pub fn read(&self) -> Option<&T> {
        self.item.last()
    }

    pub fn replace(&mut self, new_item: Option<T>) -> Option<T> {
        let item = self.item.pop();
        if let Some(new_item) = new_item {
            self.item.push(new_item);
        }

        item
    }
}
