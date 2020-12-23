use std::pin::Pin;
use std::task::{Context, Poll};

use futures::{ready, Future, Stream};
use pin_project::pin_project;

use crate::general::TCResult;
use crate::scalar::Number;
use crate::transaction::Txn;

use super::super::Coord;
use super::{Read, ReadValueAt};

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

impl<'a, S: Stream<Item = TCResult<Coord>> + 'a, T: ReadValueAt> Stream for ValueReader<'a, S, T> {
    type Item = TCResult<(Coord, Number)>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        Poll::Ready(loop {
            if let Some(mut pending) = this.pending.as_mut().as_pin_mut() {
                let result = ready!(pending.as_mut().poll(cxt));
                this.pending.set(None);
                break Some(result)
            } else if let Some(coord) = ready!(this.coords.as_mut().poll_next(cxt)) {
                match coord {
                    Ok(coord) => {
                        let read = this.access.read_value_at(&this.txn, coord);
                        this.pending.set(Some(read));
                    }
                    Err(cause) => break Some(Err(cause)),
                }
            } else {
                break None
            }
        })
    }
}
