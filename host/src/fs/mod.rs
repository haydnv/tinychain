use std::path::PathBuf;

use futures::TryFutureExt;
use futures_locks::RwLock;

use error::*;
use generic::PathSegment;
use transact::TxnId;

mod cache;
mod dir;
mod file;

pub use cache::Cache;
pub use dir::*;
pub use file::*;

#[derive(Clone)]
pub struct Root {
    cache: Cache,
}

impl Root {
    pub async fn load(mount_point: PathBuf, cache_size: usize) -> TCResult<Self> {
        // TODO: just have one global cache size
        cache::Cache::load(mount_point, cache_size)
            .map_ok(|cache| Root { cache })
            .await
    }

    pub async fn create_dir(&self, _name: PathSegment) -> TCResult<RwLock<Dir>> {
        Err(TCError::not_implemented("Root::create_dir"))
    }

    pub async fn get_dir(&self, _name: PathSegment) -> TCResult<Option<Dir>> {
        Err(TCError::not_implemented("Root::get_dir"))
    }

    pub async fn version(&self, _txn_id: TxnId) -> TCResult<RwLock<DirView>> {
        Err(TCError::not_implemented("Root::version"))
    }
}
