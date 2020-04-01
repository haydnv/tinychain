use std::collections::{BTreeMap, HashSet};
use std::hash::Hash;
use std::sync::{Arc, RwLock};

pub struct Map<K: Eq + Ord, V> {
    map: RwLock<BTreeMap<K, Arc<V>>>,
}

impl<K: Eq + Ord, V> Map<K, V> {
    pub fn new() -> Map<K, V> {
        Map {
            map: RwLock::new(BTreeMap::new()),
        }
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.map.read().unwrap().contains_key(key)
    }

    pub fn get(&self, key: &K) -> Option<Arc<V>> {
        match self.map.read().unwrap().get(key) {
            Some(state) => Some(state.clone()),
            None => None,
        }
    }

    pub fn insert(&self, key: K, value: Arc<V>) -> Option<Arc<V>> {
        self.map.write().unwrap().insert(key, value)
    }
}

pub struct Set<T: Hash> {
    set: RwLock<HashSet<T>>,
}

impl<T: Eq + Hash> Set<T> {
    pub fn new() -> Set<T> {
        Set {
            set: RwLock::new(HashSet::new()),
        }
    }

    pub fn insert(&self, value: T) -> bool {
        self.set.write().unwrap().insert(value)
    }

    pub fn contains(&self, value: &T) -> bool {
        self.set.read().unwrap().contains(value)
    }
}

pub struct Value<T: Copy> {
    val: RwLock<T>,
}

impl<T: Copy> Value<T> {
    pub fn of(value: T) -> Value<T> {
        Value {
            val: RwLock::new(value),
        }
    }

    pub fn get(&self) -> T {
        self.val.read().unwrap().clone()
    }

    pub fn set(&self, value: T) {
        *self.val.write().unwrap() = value;
    }
}
