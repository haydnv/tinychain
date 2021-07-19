//! Generic [`Stream`] types such as [`GroupStream`]

use std::fmt;
use std::mem;
use std::pin::Pin;

use async_trait::async_trait;
use destream::en;
use futures::stream::{Fuse, Stream, StreamExt, TryStreamExt};
use futures::task::{Context, Poll};
use futures::{ready, Future};
use pin_project::pin_project;

use tc_error::TCResult;
use tc_transact::fs::Dir;
use tc_transact::Transaction;
use tcgeneric::TCBoxTryStream;

#[async_trait]
pub trait Functional<'en, D: Dir>: Clone + Send + Sync + 'en {
    type Item: en::IntoStream<'en>;
    type Stream: Stream<Item = TCResult<Self::Item>> + Send + Unpin;
    type Txn: Transaction<D>;

    async fn into_stream(self, txn: Self::Txn) -> TCResult<Self::Stream>;

    async fn fold<A, Fut, F>(self, txn: Self::Txn, acc: A, op: F) -> TCResult<A>
    where
        A: Send + 'en,
        Fut: Future<Output = TCResult<A>> + Send + Unpin + 'en,
        F: Fn(A, Self::Item) -> Fut + Send + 'en,
    {
        let stream = self.into_stream(txn).await?;
        stream.try_fold(acc, op).await
    }

    async fn for_each<Fut, F>(self, txn: Self::Txn, op: F) -> TCResult<()>
    where
        Fut: Future<Output = TCResult<()>> + Send + Unpin + 'en,
        F: Fn(Self::Item) -> Fut + Send + 'en,
    {
        let stream = self.into_stream(txn).await?;
        stream.try_for_each(op).await
    }

    async fn map<M, Fut, F>(self, txn: Self::Txn, op: F) -> TCResult<TCBoxTryStream<'en, M>>
    where
        Fut: Future<Output = TCResult<M>> + Send + Unpin + 'en,
        F: Fn(Self::Item) -> Fut + Send + 'en,
    {
        let stream = self.into_stream(txn).await?;
        Ok(Box::pin(stream.and_then(op)))
    }
}

/// A [`Stream`] which groups an ordered input stream into only its unique entries using [`Eq`]
#[pin_project]
pub struct GroupStream<T, S: Stream<Item = TCResult<T>>> {
    #[pin]
    source: Fuse<S>,
    group: Option<T>,
}

impl<T: Eq + fmt::Debug, S: Stream<Item = TCResult<T>>> Stream for GroupStream<T, S> {
    type Item = TCResult<T>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        Poll::Ready(loop {
            if let Some(result) = ready!(this.source.as_mut().poll_next(cxt)) {
                match result {
                    Ok(item) => {
                        if let Some(group) = this.group {
                            if &item != group {
                                let mut new_group = item;
                                mem::swap(group, &mut new_group);
                                break Some(Ok(new_group));
                            }
                        } else {
                            *(this.group) = Some(item);
                        }
                    }
                    Err(cause) => break Some(Err(cause)),
                }
            } else {
                let mut group = None;
                mem::swap(this.group, &mut group);
                break group.map(TCResult::Ok);
            }
        })
    }
}

impl<T, S: Stream<Item = TCResult<T>>> From<S> for GroupStream<T, S> {
    fn from(source: S) -> GroupStream<T, S> {
        GroupStream {
            source: source.fuse(),
            group: None,
        }
    }
}
