use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, RwLock};

pub struct Map<K: Eq + Hash, V: Hash> {
    map: RwLock<HashMap<K, Arc<V>>>,
}

impl<K: Eq + Hash, V: Hash> Map<K, V> {
    pub fn new() -> Map<K, V> {
        Map {
            map: RwLock::new(HashMap::new()),
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
