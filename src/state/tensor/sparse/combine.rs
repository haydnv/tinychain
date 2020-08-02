use std::cmp::Ordering;
use std::mem;
use std::pin::Pin;
use std::task::{self, Poll};

use futures::stream::{Fuse, Stream, StreamExt};
use pin_project::pin_project;

use crate::value::Number;

use super::bounds::compare_coord;
use super::{SparseRow, SparseStream};

// Based on: https://github.com/rust-lang/futures-rs/blob/master/futures-util/src/stream/select.rs
#[pin_project]
pub struct SparseCombine {
    #[pin]
    left: Fuse<SparseStream>,
    #[pin]
    right: Fuse<SparseStream>,

    pending_left: Option<SparseRow>,
    pending_right: Option<SparseRow>,
}

impl SparseCombine {
    pub fn new(left: SparseStream, right: SparseStream) -> SparseCombine {
        SparseCombine {
            left: left.fuse(),
            right: right.fuse(),
            pending_left: None,
            pending_right: None,
        }
    }

    fn poll_inner(
        stream: Pin<&mut Fuse<SparseStream>>,
        pending: &mut Option<SparseRow>,
        cxt: &mut task::Context,
    ) -> bool {
        match stream.poll_next(cxt) {
            Poll::Pending => false,
            Poll::Ready(Some(row)) => {
                *pending = Some(row);
                false
            }
            Poll::Ready(None) => true,
        }
    }

    fn swap_value(pending: &mut Option<SparseRow>) -> (Vec<u64>, Number) {
        assert!(pending.is_some());

        let mut row: Option<SparseRow> = None;
        mem::swap(pending, &mut row);
        row.unwrap()
    }
}

impl Stream for SparseCombine {
    type Item = (Vec<u64>, Option<Number>, Option<Number>);

    fn poll_next(self: Pin<&mut Self>, cxt: &mut task::Context) -> Poll<Option<Self::Item>> {
        let this = self.project();

        let left_done = if this.left.is_done() {
            true
        } else {
            Self::poll_inner(this.left, this.pending_left, cxt)
        };

        let right_done = if this.right.is_done() {
            true
        } else {
            Self::poll_inner(this.right, this.pending_right, cxt)
        };

        if this.pending_left.is_some() && this.pending_right.is_some() {
            let (l_coord, _) = this.pending_left.as_ref().unwrap();
            let (r_coord, _) = this.pending_right.as_ref().unwrap();

            match compare_coord(l_coord, r_coord) {
                Ordering::Equal => {
                    let (l_coord, l_value) = Self::swap_value(this.pending_left);
                    let (_, r_value) = Self::swap_value(this.pending_right);
                    Poll::Ready(Some((l_coord, Some(l_value), Some(r_value))))
                }
                Ordering::Less => {
                    let (l_coord, l_value) = Self::swap_value(this.pending_left);
                    Poll::Ready(Some((l_coord, Some(l_value), None)))
                }
                Ordering::Greater => {
                    let (r_coord, r_value) = Self::swap_value(this.pending_right);
                    Poll::Ready(Some((r_coord, None, Some(r_value))))
                }
            }
        } else if right_done && this.pending_left.is_some() {
            let (l_coord, l_value) = Self::swap_value(this.pending_left);
            Poll::Ready(Some((l_coord, Some(l_value), None)))
        } else if left_done && this.pending_right.is_some() {
            let (r_coord, r_value) = Self::swap_value(this.pending_right);
            Poll::Ready(Some((r_coord, None, Some(r_value))))
        } else if left_done && right_done {
            Poll::Ready(None)
        } else {
            Poll::Pending
        }
    }
}
