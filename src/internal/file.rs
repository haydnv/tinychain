use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use async_trait::async_trait;
use bytes::Bytes;
use futures::Stream;

use crate::internal::block::Store;
use crate::internal::cache::Deque;
use crate::transaction::TransactionId;
use crate::value::Link;

type Blocks = Box<dyn Stream<Item = Vec<(TransactionId, Vec<Bytes>)>> + Send + Unpin>;
type FileData = (Link, Blocks);

struct SharedState {
    open: bool,
    waker: Option<Waker>,
}

pub struct FileCopier {
    contents: Deque<FileData>,
    shared_state: Arc<Mutex<SharedState>>,
}

impl FileCopier {
    pub fn new() -> FileCopier {
        FileCopier {
            contents: Deque::new(),
            shared_state: Arc::new(Mutex::new(SharedState {
                open: true,
                waker: None,
            })),
        }
    }

    pub async fn copy<T: File>(
        &mut self,
        txn_id: TransactionId,
        state: T,
        dest: Arc<Store>,
    ) -> Arc<T> {
        state.copy_file(txn_id, self).await;
        self.end();
        T::from_file(self, dest).await
    }

    pub fn end(&mut self) {
        self.shared_state.lock().unwrap().open = false;
    }

    pub fn write_file(&mut self, path: Link, blocks: Blocks) {
        let shared_state = self.shared_state.lock().unwrap();
        if !shared_state.open {
            panic!("Tried to write file to closed FileCopier");
        } else if path.len() != 1 {
            panic!("Tried to write file in subdirectory: {}", path);
        }

        self.contents.push_back((path, blocks));
        if let Some(waker) = &shared_state.waker {
            waker.clone().wake();
        }
    }
}

impl Stream for FileCopier {
    type Item = FileData;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context) -> Poll<Option<Self::Item>> {
        let mut shared_state = self.shared_state.lock().unwrap();
        if self.contents.is_empty() {
            if shared_state.open {
                shared_state.waker = Some(cxt.waker().clone());
                Poll::Pending
            } else {
                Poll::Ready(None)
            }
        } else {
            Poll::Ready(self.contents.pop_front())
        }
    }
}

#[async_trait]
pub trait File {
    async fn copy_file(&self, txn_id: TransactionId, copier: &mut FileCopier);

    async fn from_file(copier: &mut FileCopier, dest: Arc<Store>) -> Arc<Self>;
}

#[async_trait]
impl<T: File + Sync + Send> File for Arc<T> {
    async fn copy_file(&self, txn_id: TransactionId, copier: &mut FileCopier) {
        self.copy_file(txn_id, copier).await
    }

    async fn from_file(copier: &mut FileCopier, dest: Arc<Store>) -> Arc<Self> {
        Self::from_file(copier, dest).await
    }
}
