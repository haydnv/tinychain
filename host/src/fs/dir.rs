use std::collections::HashMap;

use transact::TxnId;

use super::cache::CacheDir;

pub struct Dir {
    dir: CacheDir,
    versions: HashMap<TxnId, CacheDir>,
}
