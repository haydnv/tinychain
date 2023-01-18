//! Generic [`Stream`] types such as [`GroupStream`]

use std::fmt;
use std::mem;
use std::pin::Pin;

use async_trait::async_trait;
use futures::ready;
use futures::stream::{Fuse, Stream, StreamExt, TryStreamExt};
use futures::task::{Context, Poll};
use pin_project::pin_project;
use safecast::TryCastFrom;

use tc_error::*;
use tc_value::Value;
use tcgeneric::TCBoxTryStream;

use crate::state::State;
use crate::txn::Txn;

use super::{Source, TCStream};

#[derive(Clone)]
pub struct Aggregate {
    source: TCStream,
}

impl Aggregate {
    pub fn new(source: TCStream) -> Self {
        Self { source }
    }
}

#[async_trait]
impl Source for Aggregate {
    async fn into_stream(self, txn: Txn) -> TCResult<TCBoxTryStream<'static, State>> {
        let source = self.source.into_stream(txn).await?;

        let values = source.map(|r| {
            r.and_then(|state| {
                Value::try_cast_from(state, |s| {
                    bad_request!("to aggregate a Stream requires a Value, not {}", s)
                })
            })
        });

        let aggregate: TCBoxTryStream<'static, State> =
            Box::pin(GroupStream::from(values).map_ok(State::from));

        Ok(aggregate)
    }
}

impl From<Aggregate> for TCStream {
    fn from(aggregate: Aggregate) -> Self {
        TCStream::Aggregate(Box::new(aggregate))
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
