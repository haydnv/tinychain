use std::mem;
use std::pin::Pin;
use std::task::{self, ready};

use futures::stream::{Fuse, FusedStream, Stream};
use futures::StreamExt;
use ha_ndarray::{ArrayBase, CType, NDArrayRead, Shape};
use itertools::MultiProduct;
use pin_project::pin_project;

use tc_error::*;

use crate::tensor::shape::AxisRangeIter;
use crate::tensor::{autoqueue, Coord, Range};

#[pin_project]
pub struct BlockResize<S, T> {
    #[pin]
    source: Fuse<S>,
    shape: Shape,
    pending: Vec<T>,
}

impl<S, T> BlockResize<S, T>
where
    S: Stream,
{
    pub fn new(source: S, block_shape: Shape) -> TCResult<Self> {
        let size = block_shape.iter().product::<usize>();

        Ok(Self {
            source: source.fuse(),
            shape: block_shape,
            pending: Vec::with_capacity(size * 2),
        })
    }
}

impl<S, A, T> Stream for BlockResize<S, T>
where
    S: Stream<Item = Result<A, TCError>>,
    A: NDArrayRead<DType = T>,
    T: CType,
{
    type Item = Result<ArrayBase<Vec<T>>, TCError>;

    fn poll_next(
        self: Pin<&mut Self>,
        cxt: &mut task::Context<'_>,
    ) -> task::Poll<Option<Self::Item>> {
        let shape = self.shape.to_vec();
        let size = shape.iter().product::<usize>();
        let mut this = self.project();

        task::Poll::Ready(loop {
            if this.pending.len() > size {
                debug_assert_eq!(shape.iter().product::<usize>(), size);
                let data = this.pending.drain(..size).collect();
                let data = ArrayBase::<Vec<T>>::new(shape, data).map_err(TCError::from);
                break Some(data);
            } else {
                match ready!(this.source.as_mut().poll_next(cxt)) {
                    Some(Ok(block)) => match autoqueue(&block) {
                        Ok(queue) => match block.read(&queue) {
                            Ok(buffer) => match buffer.to_slice() {
                                Ok(slice) => this.pending.extend(slice.as_ref()),
                                Err(cause) => break Some(Err(TCError::from(cause))),
                            },
                            Err(cause) => break Some(Err(TCError::from(cause))),
                        },
                        Err(cause) => break Some(Err(TCError::from(cause))),
                    },
                    Some(Err(cause)) => break Some(Err(cause)),
                    None if this.pending.is_empty() => break None,
                    None => {
                        let mut shape = shape;
                        let trailing_size = shape.iter().skip(1).product::<usize>();
                        shape[0] = this.pending.len() / trailing_size;
                        debug_assert_eq!(this.pending.len() % trailing_size, 0);

                        let mut data = vec![];
                        mem::swap(this.pending, &mut data);

                        let data = ArrayBase::<Vec<T>>::new(shape, data).map_err(TCError::from);
                        break Some(data);
                    }
                }
            }
        })
    }
}

#[pin_project]
pub struct ValueStream<S, T> {
    #[pin]
    filled: Fuse<S>,

    affected: MultiProduct<AxisRangeIter>,
    next_coord: Option<Coord>,
    next_filled: Option<(Coord, T)>,
    zero: T,
}

impl<'a, S: StreamExt + 'a, T: Copy + 'a> ValueStream<S, T> {
    pub fn new(filled: S, range: Range, zero: T) -> Self {
        let mut affected = range.affected();
        let next_coord = affected.next().map(Coord::from);

        Self {
            filled: filled.fuse(),
            affected,
            next_coord,
            next_filled: None,
            zero,
        }
    }
}

impl<S: Stream<Item = TCResult<(Coord, T)>>, T: Copy> Stream for ValueStream<S, T> {
    type Item = TCResult<T>;

    fn poll_next(
        self: Pin<&mut Self>,
        cxt: &mut task::Context<'_>,
    ) -> task::Poll<Option<Self::Item>> {
        let mut this = self.project();

        task::Poll::Ready(loop {
            let next_coord = if let Some(next_coord) = this.next_coord {
                next_coord
            } else {
                break None;
            };

            let mut next = None;
            mem::swap(this.next_filled, &mut next);
            if let Some((filled_coord, value)) = next {
                break if next_coord == &filled_coord {
                    *(this.next_coord) = this.affected.next().map(Coord::from);
                    Some(Ok(value))
                } else {
                    *(this.next_coord) = this.affected.next().map(Coord::from);
                    *(this.next_filled) = Some((filled_coord, value));
                    Some(Ok(*this.zero))
                };
            } else if this.filled.is_terminated() {
                *(this.next_coord) = this.affected.next().map(Coord::from);
                break Some(Ok(*this.zero));
            } else {
                match ready!(this.filled.as_mut().poll_next(cxt)) {
                    Some(Ok((filled_coord, value))) => {
                        break if next_coord == &filled_coord {
                            *(this.next_coord) = this.affected.next().map(Coord::from);
                            Some(Ok(value))
                        } else {
                            *(this.next_coord) = this.affected.next().map(Coord::from);
                            *(this.next_filled) = Some((filled_coord, value));
                            Some(Ok(*this.zero))
                        };
                    }
                    None => {
                        *(this.next_coord) = this.affected.next().map(Coord::from);
                        break Some(Ok(*this.zero));
                    }
                    Some(Err(cause)) => {
                        *(this.next_coord) = this.affected.next().map(Coord::from);
                        break Some(Err(cause));
                    }
                }
            }
        })
    }
}
