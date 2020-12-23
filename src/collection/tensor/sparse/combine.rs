use std::cmp::Ordering;
use std::mem;
use std::pin::Pin;
use std::task::{self, Poll};

use futures::stream::{Fuse, Stream, StreamExt};
use pin_project::pin_project;

use crate::general::TCResult;
use crate::scalar::Number;

use super::super::Coord;
use super::super::bounds::Shape;
use super::{SparseRow, SparseStream};

// Based on: https://github.com/rust-lang/futures-rs/blob/master/futures-util/src/stream/select.rs
#[pin_project]
pub struct SparseCombine<'a> {
    #[pin]
    left: Fuse<SparseStream<'a>>,
    #[pin]
    right: Fuse<SparseStream<'a>>,

    coord_bounds: Vec<u64>,

    pending_left: Option<(u64, SparseRow)>,
    pending_right: Option<(u64, SparseRow)>,
}

impl<'a> SparseCombine<'a> {
    pub fn new(
        shape: &Shape,
        left: SparseStream<'a>,
        right: SparseStream<'a>,
    ) -> SparseCombine<'a> {
        let coord_bounds = (0..shape.len())
            .map(|axis| shape[axis + 1..].iter().product())
            .collect();

        SparseCombine {
            left: left.fuse(),
            right: right.fuse(),
            coord_bounds,
            pending_left: None,
            pending_right: None,
        }
    }

    fn poll_inner(
        stream: Pin<&mut Fuse<SparseStream<'a>>>,
        coord_bounds: &[u64],
        pending: &mut Option<(u64, SparseRow)>,
        cxt: &mut task::Context,
    ) -> TCResult<bool> {
        match stream.poll_next(cxt) {
            Poll::Pending => Ok(false),
            Poll::Ready(Some(Ok((coord, value)))) => {
                let offset = coord_to_offset(&coord, coord_bounds);
                *pending = Some((offset, (coord, value)));
                Ok(false)
            }
            Poll::Ready(Some(Err(cause))) => Err(cause),
            Poll::Ready(None) => Ok(true),
        }
    }

    fn swap_value(pending: &mut Option<(u64, SparseRow)>) -> SparseRow {
        assert!(pending.is_some());

        let mut row: Option<(u64, SparseRow)> = None;
        mem::swap(pending, &mut row);
        let (_, row) = row.unwrap();
        row
    }
}

impl<'a> Stream for SparseCombine<'a> {
    type Item = TCResult<(Coord, Option<Number>, Option<Number>)>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut task::Context) -> Poll<Option<Self::Item>> {
        let this = self.project();

        let left_done = if this.left.is_done() {
            true
        } else {
            match Self::poll_inner(this.left, this.coord_bounds, this.pending_left, cxt) {
                Err(cause) => return Poll::Ready(Some(Err(cause))),
                Ok(done) => done,
            }
        };

        let right_done = if this.right.is_done() {
            true
        } else {
            match Self::poll_inner(this.right, this.coord_bounds, this.pending_right, cxt) {
                Err(cause) => return Poll::Ready(Some(Err(cause))),
                Ok(done) => done,
            }
        };

        if this.pending_left.is_some() && this.pending_right.is_some() {
            let (l_offset, _) = this.pending_left.as_ref().unwrap();
            let (r_offset, _) = this.pending_right.as_ref().unwrap();

            match l_offset.cmp(r_offset) {
                Ordering::Equal => {
                    let (l_coord, l_value) = Self::swap_value(this.pending_left);
                    let (_, r_value) = Self::swap_value(this.pending_right);
                    Poll::Ready(Some(Ok((l_coord, Some(l_value), Some(r_value)))))
                }
                Ordering::Less => {
                    let (l_coord, l_value) = Self::swap_value(this.pending_left);
                    Poll::Ready(Some(Ok((l_coord, Some(l_value), None))))
                }
                Ordering::Greater => {
                    let (r_coord, r_value) = Self::swap_value(this.pending_right);
                    Poll::Ready(Some(Ok((r_coord, None, Some(r_value)))))
                }
            }
        } else if right_done && this.pending_left.is_some() {
            let (l_coord, l_value) = Self::swap_value(this.pending_left);
            Poll::Ready(Some(Ok((l_coord, Some(l_value), None))))
        } else if left_done && this.pending_right.is_some() {
            let (r_coord, r_value) = Self::swap_value(this.pending_right);
            Poll::Ready(Some(Ok((r_coord, None, Some(r_value)))))
        } else if left_done && right_done {
            Poll::Ready(None)
        } else {
            Poll::Pending
        }
    }
}

fn coord_to_offset(coord: &[u64], coord_bounds: &[u64]) -> u64 {
    coord_bounds
        .iter()
        .zip(coord.iter())
        .map(|(d, x)| d * x)
        .sum()
}
