use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures::Stream;

use crate::internal::cache::Deque;
use crate::internal::FsDir;
use crate::value::{Link, TCResult};

type FileData = (Link, Box<dyn Stream<Item = Bytes>>);

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

    pub fn write_file(&mut self, path: Link, blocks: Box<dyn Stream<Item = Bytes>>) {
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

pub trait File {
    fn create(fs_dir: Arc<FsDir>) -> Arc<Self>;

    fn copy(reader: FileReader, dest: Arc<FsDir>) -> TCResult<Arc<Self>>;

    fn into(&self, writer: FileWriter);
}
