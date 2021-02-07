use std::collections::HashSet;
use std::fs::Metadata;
use std::path::PathBuf;

use bytes::Bytes;
use futures::future::TryFutureExt;
use tokio::fs;

use error::*;
use generic::PathSegment;

use super::{file_name, fs_path, io_err};

pub struct File {
    mount_point: PathBuf,
    contents: HashSet<PathSegment>,
}

impl File {
    pub fn load(mount_point: PathBuf, entries: Vec<(fs::DirEntry, Metadata)>) -> TCResult<Self> {
        let contents = if entries.iter().all(|(_, meta)| meta.is_file()) {
            let mut contents = HashSet::new();
            for (handle, _) in entries.into_iter() {
                contents.insert(file_name(&handle)?);
            }
            contents
        } else {
            return Err(TCError::internal(format!(
                "host directory {:?} contains both blocks and subdirectories",
                mount_point
            )));
        };

        Ok(Self {
            mount_point,
            contents,
        })
    }

    pub fn create(mount_point: PathBuf) -> Self {
        Self {
            mount_point,
            contents: HashSet::new(),
        }
    }

    pub fn block_ids(&'_ self) -> &'_ HashSet<PathSegment> {
        &self.contents
    }

    pub async fn create_block(&mut self, name: PathSegment, initial_value: Bytes) -> TCResult<()> {
        if self.contents.contains(&name) {
            return Err(TCError::bad_request(
                "There is already a block with ID",
                &name,
            ));
        }

        let block_path = fs_path(&self.mount_point, &name);
        fs::write(&block_path, initial_value)
            .map_err(|e| io_err(e, &block_path))
            .await?;

        self.contents.insert(name);
        Ok(())
    }

    pub async fn copy_all(&mut self, source: &Self) -> TCResult<()> {
        for block_id in source.block_ids().clone().into_iter() {
            self.copy_block(block_id, source).await?;
        }

        Ok(())
    }

    pub async fn copy_block(&mut self, name: PathSegment, source: &Self) -> TCResult<()> {
        let block = source
            .get_block(&name)
            .await?
            .ok_or_else(|| TCError::not_found(&name))?;

        let path = fs_path(&self.mount_point, &name);
        fs::write(&path, &block)
            .map_err(|e| io_err(e, &path))
            .await?;

        self.contents.insert(name);
        Ok(())
    }

    pub async fn delete(&mut self, name: &PathSegment) -> TCResult<()> {
        if self.contents.contains(name) {
            let path = fs_path(&self.mount_point, name);
            fs::remove_file(&path).map_err(|e| io_err(e, &path)).await?;
            self.contents.remove(name);
            Ok(())
        } else {
            Err(TCError::not_found(name))
        }
    }

    pub async fn get_block(&self, name: &PathSegment) -> TCResult<Option<Bytes>> {
        if self.contents.contains(name) {
            let path = fs_path(&self.mount_point, name);

            fs::read(&path)
                .map_ok(Bytes::from)
                .map_ok(Some)
                .map_err(|e| io_err(e, &path))
                .await
        } else {
            Err(TCError::not_found(name))
        }
    }

    pub fn is_empty(&self) -> bool {
        self.contents.is_empty()
    }
}
