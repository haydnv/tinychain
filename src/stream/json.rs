use std::mem;
use std::pin::Pin;
use std::task::{self, Poll};

use futures::stream::{Fuse, Stream, StreamExt};
use pin_project::pin_project;
use serde::Serialize;

use crate::error;
use crate::general::TCResult;

#[pin_project]
pub struct JsonListStream<I: Serialize, S: Stream<Item = I>> {
    #[pin]
    source: Fuse<S>,

    started: bool,
    next: Option<TCResult<String>>,
}

impl<'a, I: Serialize, S: Stream<Item = I> + 'a> Stream for JsonListStream<I, S>
where
    S: 'a,
{
    type Item = TCResult<String>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut task::Context) -> Poll<Option<Self::Item>> {
        let this = self.project();

        match this.source.poll_next(cxt) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Some(value)) if this.next.is_none() => {
                *this.started = true;

                let mut json =
                    Some(serde_json::to_string_pretty(&value).map_err(error::TCError::from));
                mem::swap(this.next, &mut json);
                Poll::Ready(Some(Ok("[".to_string())))
            }
            Poll::Ready(Some(value)) => {
                *this.started = true;

                let mut json =
                    Some(serde_json::to_string_pretty(&value).map_err(error::TCError::from));
                mem::swap(this.next, &mut json);

                if let Some(Ok(val)) = json {
                    Poll::Ready(Some(Ok(format!("{}, ", val))))
                } else {
                    Poll::Ready(json)
                }
            }
            Poll::Ready(None) if this.next.is_some() => {
                let mut json = None;
                mem::swap(this.next, &mut json);

                if let Some(Ok(val)) = json {
                    Poll::Ready(Some(Ok(format!("{}]", val))))
                } else {
                    Poll::Ready(json)
                }
            }
            Poll::Ready(None) if !*this.started => {
                *this.started = true;
                Poll::Ready(Some(Ok("[]".to_string())))
            }
            Poll::Ready(None) => Poll::Ready(None),
        }
    }
}

impl<I: Serialize, S: Stream<Item = I>> From<S> for JsonListStream<I, S> {
    fn from(s: S) -> JsonListStream<I, S> {
        JsonListStream {
            source: s.fuse(),
            started: false,
            next: None,
        }
    }
}
