use std::collections::VecDeque;

use futures::stream::{self, Stream, StreamExt};

use crate::collection::{Collection, CollectionInstance};
use crate::general::TCResult;
use crate::scalar::Scalar;
use crate::transaction::Txn;

pub struct StreamBuffer {
    // TODO: use the filesystem to buffer
    buffer: VecDeque<Scalar>,
}

impl StreamBuffer {
    pub async fn new(state: Collection, txn: Txn) -> TCResult<Self> {
        let mut buffer = VecDeque::new();
        let mut stream = state.to_stream(&txn).await?;
        while let Some(scalar) = stream.next().await {
            buffer.push_back(scalar);
        }

        Ok(Self { buffer })
    }

    pub fn into_stream(self) -> impl Stream<Item = Scalar> + Send + Sync + Unpin {
        stream::iter(self.buffer)
    }
}
