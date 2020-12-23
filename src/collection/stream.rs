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

impl<T: Eq + Clone, S: Stream<Item = TCResult<T>>> Stream for GroupStream<T, S> {
    type Item = TCResult<T>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        let source = this.source.as_mut().get_pin_mut();
        let item = match ready!(source.poll_next(cxt)) {
            Some(Ok(item)) => item,
            None => return Poll::Ready(None),
            Some(Err(cause)) => return Poll::Ready(Some(Err(cause))),
        };

        let skip = if let Some(group) = this.group {
            group == &item
        } else {
            false
        };

        if skip {
            Poll::Pending
        } else {
            *(this.group) = Some(item.clone());
            Poll::Ready(Some(Ok(item)))
        }
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
