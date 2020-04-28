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

pub struct FileWriter {
    open: bool,
    contents: Deque<FileData>,
}

impl FileWriter {
    pub fn new() -> FileWriter {
        FileWriter {
            open: true,
            contents: Deque::new(),
        }
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

pub struct FileReader {
    source: FileWriter,
}

impl Stream for FileReader {
    type Item = FileData;

    fn poll_next(self: Pin<&mut Self>, _cxt: &mut Context) -> Poll<Option<Self::Item>> {
        if !self.source.open {
            Poll::Ready(None)
        } else if self.source.open && self.source.contents.is_empty() {
            Poll::Pending
        } else {
            Poll::Ready(self.source.contents.pop_front())
        }
    }
}

#[async_trait]
pub trait File {
    async fn copy_from(reader: &mut FileReader, dest: Arc<FsDir>) -> Arc<Self>;

    async fn copy_to(&self, txn_id: TransactionId, writer: &mut FileWriter);
}

#[async_trait]
impl<T: File + Sync + Send> File for Arc<T> {
    async fn copy_from(reader: &mut FileReader, dest: Arc<FsDir>) -> Arc<Self> {
        Self::copy_from(reader, dest).await
    }

    async fn copy_to(&self, txn_id: TransactionId, writer: &mut FileWriter) {
        self.copy_to(txn_id, writer).await
    }
}

pub async fn copy(_txn_id: TransactionId, _state: impl File, _context: Arc<FsDir>) {
    // TODO
}
