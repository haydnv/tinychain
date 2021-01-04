use std::fmt;
use std::mem;
use std::pin::Pin;

use futures::ready;
use futures::stream::{Fuse, Stream, StreamExt};
use futures::task::{Context, Poll};
use pin_project::pin_project;

use crate::TCResult;

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
