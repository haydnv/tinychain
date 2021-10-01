use std::mem;
use std::pin::Pin;

use futures::ready;
use futures::stream::{Fuse, FusedStream, Stream, StreamExt};
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

    affected: Coords,
    next_coord: Option<Coord>,
    next_filled: Option<(Coord, Number)>,
    zero: Number,
}

impl<'a, S: StreamExt + 'a> SparseValueStream<S> {
    pub async fn new(filled: S, bounds: Bounds, zero: Number) -> TCResult<Self> {
        let mut affected = bounds.affected();
        let next_coord = affected.next();

        Ok(Self {
            filled: filled.fuse(),
            affected,
            next_coord,
            next_filled: None,
            zero,
        })
    }
}

impl<S: Stream<Item = TCResult<(Coord, Number)>>> Stream for SparseValueStream<S> {
    type Item = TCResult<Number>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        Poll::Ready(loop {
            let next_coord = if let Some(next_coord) = this.next_coord {
                next_coord
            } else {
                break None;
            };

            let mut next = None;
            mem::swap(this.next_filled, &mut next);
            if let Some((filled_coord, value)) = next {
                break if next_coord == &filled_coord {
                    *(this.next_coord) = this.affected.next();
                    Some(Ok(value))
                } else {
                    *(this.next_coord) = this.affected.next();
                    *(this.next_filled) = Some((filled_coord, value));
                    Some(Ok(*this.zero))
                };
            } else if this.filled.is_terminated() {
                *(this.next_coord) = this.affected.next();
                break Some(Ok(*this.zero));
            } else {
                match ready!(this.filled.as_mut().poll_next(cxt)) {
                    Some(Ok((filled_coord, value))) => {
                        break if next_coord == &filled_coord {
                            *(this.next_coord) = this.affected.next();
                            Some(Ok(value))
                        } else {
                            *(this.next_coord) = this.affected.next();
                            *(this.next_filled) = Some((filled_coord, value));
                            Some(Ok(*this.zero))
                        };
                    }
                    None => {
                        *(this.next_coord) = this.affected.next();
                        break Some(Ok(*this.zero));
                    }
                    Some(Err(cause)) => {
                        *(this.next_coord) = this.affected.next();
                        break Some(Err(cause));
                    }
                }
            }
        })
    }
}
