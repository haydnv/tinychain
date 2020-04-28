use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use async_trait::async_trait;
use bytes::Bytes;
use futures::Stream;

use crate::internal::cache::Deque;
use crate::internal::FsDir;
use crate::transaction::TransactionId;
use crate::value::Link;

type Blocks = Box<dyn Stream<Item = Vec<(TransactionId, Vec<Bytes>)>> + Send + Unpin>;
type FileData = (Link, Blocks);

pub struct FileCopier {
    open: bool,
    contents: Deque<FileData>,
}

impl FileCopier {
    pub fn new() -> FileCopier {
        FileCopier {
            open: true,
            contents: Deque::new(),
        }
    }

    pub async fn copy<T: File>(
        &mut self,
        txn_id: TransactionId,
        state: T,
        dest: Arc<FsDir>,
    ) -> Arc<T> {
        state.copy_file(txn_id, self).await;
        T::from_file(self, dest).await
    }

    pub fn end(&mut self) {
        self.open = false
    }

    pub fn write_file(&mut self, path: Link, blocks: Blocks) {
        if path.len() != 1 {
            panic!("Tried to write file in subdirectory: {}", path);
        }

        self.contents.push_back((path, blocks))
    }
}

impl Stream for FileCopier {
    type Item = FileData;

    fn poll_next(self: Pin<&mut Self>, _cxt: &mut Context) -> Poll<Option<Self::Item>> {
        if !self.open {
            Poll::Ready(None)
        } else if self.open && self.contents.is_empty() {
            Poll::Pending
        } else {
            Poll::Ready(self.contents.pop_front())
        }
    }
}

#[async_trait]
pub trait File {
    async fn copy_file(&self, txn_id: TransactionId, copier: &mut FileCopier);

    async fn from_file(copier: &mut FileCopier, dest: Arc<FsDir>) -> Arc<Self>;
}

#[async_trait]
impl<T: File + Sync + Send> File for Arc<T> {
    async fn copy_file(&self, txn_id: TransactionId, copier: &mut FileCopier) {
        self.copy_file(txn_id, copier).await
    }

    async fn from_file(copier: &mut FileCopier, dest: Arc<FsDir>) -> Arc<Self> {
        Self::from_file(copier, dest).await
    }
}
