use std::mem;
use std::pin::Pin;

use futures::ready;
use futures::stream::{Fuse, Stream, StreamExt};
use futures::task::{Context, Poll};
use pin_project::pin_project;

use tc_error::*;
use tc_value::Number;

use crate::bounds::{Bounds, Coords};
use crate::Coord;

#[pin_project]
pub struct SparseValueStream<S> {
    #[pin]
    filled: Fuse<S>,

    coords: Coords,
    next: Option<(Coord, Number)>,
    zero: Number,
}

impl<'a, S: StreamExt + 'a> SparseValueStream<S> {
    pub async fn new(filled: S, bounds: Bounds, zero: Number) -> TCResult<Self> {
        let coords = bounds.affected();

        Ok(Self {
            filled: filled.fuse(),
            coords,
            next: None,
            zero,
        })
    }
}

impl<S: Stream<Item = TCResult<(Coord, Number)>>> Stream for SparseValueStream<S> {
    type Item = TCResult<Number>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        Poll::Ready(loop {
            let next_coord = match this.coords.next() {
                Some(coord) => coord,
                None => break None,
            };

            let mut next = None;
            mem::swap(&mut next, this.next);
            if let Some((filled_coord, value)) = next {
                break if next_coord == filled_coord {
                    Some(Ok(value))
                } else {
                    Some(Ok(*this.zero))
                };
            } else {
                match ready!(this.filled.as_mut().poll_next(cxt)) {
                    Some(Ok((filled_coord, value))) => {
                        break if next_coord == filled_coord {
                            Some(Ok(value))
                        } else {
                            *(this.next) = Some((filled_coord, value));
                            Some(Ok(*this.zero))
                        }
                    }
                    None => break Some(Ok(*this.zero)),
                    Some(Err(cause)) => break Some(Err(cause)),
                }
            }
        })
    }
}
