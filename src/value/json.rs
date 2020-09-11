use std::mem;
use std::pin::Pin;
use std::task::{self, Poll};

use futures::stream::{Fuse, Stream, StreamExt};
use pin_project::pin_project;

use crate::class::TCResult;
use crate::error;

use super::Value;

#[pin_project]
pub struct JsonListStream<S: Stream<Item = Value>> {
    #[pin]
    source: Fuse<S>,

    next: Option<TCResult<String>>,
}

impl<S: Stream<Item = Value>> Stream for JsonListStream<S> {
    type Item = TCResult<String>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut task::Context) -> Poll<Option<Self::Item>> {
        let this = self.project();

        match this.source.poll_next(cxt) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Some(value)) if this.next.is_none() => {
                let mut json =
                    Some(serde_json::to_string_pretty(&value).map_err(error::TCError::from));
                mem::swap(this.next, &mut json);
                Poll::Ready(Some(Ok("[".to_string())))
            }
            Poll::Ready(Some(value)) => {
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
            Poll::Ready(None) => Poll::Ready(None),
        }
    }
}

impl<S: Stream<Item = Value>> From<S> for JsonListStream<S> {
    fn from(s: S) -> JsonListStream<S> {
        JsonListStream {
            source: s.fuse(),
            next: None,
        }
    }
}
