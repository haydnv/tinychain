use std::collections::HashSet;
use std::convert::TryFrom;
use std::marker::PhantomData;
use std::path::PathBuf;

use uplock::*;

use tc_error::*;

use super::*;

#[derive(Clone)]
pub struct File<B> {
    path: PathBuf,
    cache: Cache,
    contents: RwLock<HashSet<BlockId>>,
    phantom: PhantomData<B>,
}

impl<B: BlockData> File<B> {
    pub fn create(path: PathBuf, cache: Cache) -> Self {
        File {
            path,
            cache,
            contents: RwLock::new(HashSet::new()),
            phantom: PhantomData,
        }
    }
}

impl<B: BlockData> File<B>
where
    CacheBlock: From<CacheLock<B>>,
    CacheLock<B>: TryFrom<CacheBlock, Error = TCError>,
{
    pub async fn contains_block(&self, block_id: &BlockId) -> bool {
        let contents = self.contents.read().await;
        contents.contains(block_id)
    }

    pub async fn create_block(&self, block_id: BlockId, block: B) -> TCResult<CacheLock<B>> {
        let mut contents = self.contents.write().await;
        if contents.contains(&block_id) {
            return Err(TCError::bad_request(
                "there is already a block with this ID",
                block_id,
            ));
        }

        let path = fs_path(&self.path, &block_id);
        contents.insert(block_id);
        self.cache.write(path, block).await
    }

    pub async fn get_block(&self, block_id: &BlockId) -> TCResult<Option<CacheLock<B>>> {
        let path = fs_path(&self.path, block_id);
        self.cache.read(&path).await
    }
}
