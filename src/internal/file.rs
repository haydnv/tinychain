use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use async_trait::async_trait;
use bytes::Bytes;
use futures::Stream;

use crate::internal::block::{Block, Store};
use crate::internal::cache::Deque;
use crate::transaction::TransactionId;
use crate::value::TCPath;

type Blocks = Box<dyn Stream<Item = Bytes> + Send + Unpin>;
type FileData = (TCPath, Blocks);

#[async_trait]
pub trait File {
    type Block: Block;

    async fn copy_from(
        reader: &mut FileCopier,
        txn_id: &TransactionId,
        dest: Arc<Store>,
    ) -> Arc<Self>;

    async fn copy_into(&self, txn_id: TransactionId, writer: &mut FileCopier);

    async fn from_store(txn_id: &TransactionId, store: Arc<Store>) -> Arc<Self>;
}

struct SharedState {
    open: bool,
    waker: Option<Waker>,
}

pub struct FileCopier {
    contents: Deque<FileData>,
    shared_state: Arc<Mutex<SharedState>>,
}

impl FileCopier {
    pub fn open() -> FileCopier {
        FileCopier {
            contents: Deque::new(),
            shared_state: Arc::new(Mutex::new(SharedState {
                open: true,
                waker: None,
            })),
        }
    }

    pub async fn copy<F: File>(txn_id: TransactionId, state: &F, dest: Arc<Store>) -> Arc<F> {
        let mut copier = Self::open();
        state.copy_into(txn_id.clone(), &mut copier).await;
        copier.close();
        F::copy_from(&mut copier, &txn_id, dest).await
    }

    pub fn close(&mut self) {
        self.shared_state.lock().unwrap().open = false;
    }

    pub fn write_file(&mut self, path: TCPath, blocks: Blocks) {
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
