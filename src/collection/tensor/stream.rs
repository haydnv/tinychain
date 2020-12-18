use std::pin::Pin;
use std::task::{Context, Poll};

use futures::{ready, Future, Stream};
use pin_project::pin_project;

use crate::general::TCResult;
use crate::scalar::Number;
use crate::transaction::Txn;

pub type Read<'a> = Pin<Box<dyn Future<Output = TCResult<(Vec<u64>, Number)>> + Send + 'a>>;

pub trait ReadValueAt {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Vec<u64>) -> Read<'a>;
}

#[pin_project]
pub struct ValueReader<'a, S: Stream + 'a, T> {
    #[pin]
    coords: Pin<Box<S>>,

    #[pin]
    pending: Option<Read<'a>>,

    txn: &'a Txn,
    access: &'a T,
}

impl<'a, S: Stream + 'a, T> ValueReader<'a, S, T> {
    pub fn new(coords: S, txn: &'a Txn, access: &'a T) -> Self {
        Self {
            coords: Box::pin(coords),
            txn,
            access,
            pending: None,
        }
    }
}

impl<'a, S: Stream<Item = TCResult<Vec<u64>>> + 'a, T: ReadValueAt> Stream
    for ValueReader<'a, S, T>
{
    type Item = TCResult<(Vec<u64>, Number)>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        if let Some(mut pending) = this.pending.as_mut().as_pin_mut() {
            let result = ready!(pending.as_mut().poll(cxt));
            this.pending.set(None);
            Poll::Ready(Some(result))
        } else if let Some(coord) = ready!(this.coords.as_mut().poll_next(cxt)) {
            match coord {
                Ok(coord) => {
                    this.pending
                        .set(Some(this.access.read_value_at(&this.txn, coord)));
                    Poll::Pending
                }
                Err(cause) => Poll::Ready(Some(Err(cause))),
            }
        } else {
            Poll::Ready(None)
        }
    }
}
