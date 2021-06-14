use std::pin::Pin;
use std::task::{Context, Poll};

use futures::{ready, Future, Stream};
use pin_project::pin_project;

use tc_error::*;
use tc_transact::fs::Dir;
use tc_transact::Transaction;
use tc_value::Number;

use crate::Coord;

use super::{Read, ReadValueAt};
use std::marker::PhantomData;

#[pin_project]
pub struct ValueReader<'a, S: Stream + 'a, D, T, A> {
    #[pin]
    coords: Pin<Box<S>>,

    #[pin]
    pending: Option<Read<'a>>,

    txn: T,
    access: A,

    dir: PhantomData<D>,
}

impl<'a, S: Stream + 'a, D, T, A> ValueReader<'a, S, D, T, A> {
    pub fn new(coords: S, txn: T, access: A) -> Self {
        Self {
            coords: Box::pin(coords),
            txn,
            access,
            pending: None,

            dir: PhantomData,
        }
    }
}

impl<'a, S, D, T, A> Stream for ValueReader<'a, S, D, T, A>
where
    S: Stream<Item = TCResult<Coord>> + 'a,
    D: Dir,
    T: Transaction<D>,
    A: Clone + ReadValueAt<D, Txn = T>,
{
    type Item = TCResult<(Coord, Number)>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        Poll::Ready(loop {
            if let Some(mut pending) = this.pending.as_mut().as_pin_mut() {
                let result = ready!(pending.as_mut().poll(cxt));
                this.pending.set(None);
                break Some(result);
            } else if let Some(coord) = ready!(this.coords.as_mut().poll_next(cxt)) {
                match coord {
                    Ok(coord) => {
                        let read = this.access.clone().read_value_at(this.txn.clone(), coord);
                        this.pending.set(Some(read));
                    }
                    Err(cause) => break Some(Err(cause)),
                }
            } else {
                break None;
            }
        })
    }
}
