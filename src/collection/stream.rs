use std::fmt;
use std::pin::Pin;

use futures::ready;
use futures::stream::{Fuse, Stream, StreamExt};
use pin_project::pin_project;

use crate::general::TCResult;
use futures::task::{Context, Poll};

#[pin_project]
pub struct GroupStream<T, S: Stream<Item = TCResult<T>>> {
    #[pin]
    source: Fuse<S>,

    group: Option<T>,
}

impl<T: Eq + Clone + fmt::Debug, S: Stream<Item = TCResult<T>>> Stream for GroupStream<T, S> {
    type Item = TCResult<T>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        Poll::Ready(loop {
            if let Some(result) = ready!(this.source.as_mut().poll_next(cxt)) {
                match result {
                    Ok(item) => {
                        let skip = if let Some(group) = this.group {
                            &item == group
                        } else {
                            false
                        };

                        if !skip {
                            *(this.group) = Some(item.clone());
                            break Some(Ok(item));
                        }
                    }
                    Err(cause) => break Some(Err(cause)),
                }
            } else {
                break None;
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
