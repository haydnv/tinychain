use std::cmp::Ordering;
use std::mem;
use std::pin::Pin;
use std::task::{self, Poll};

use futures::stream::{Fuse, Stream, StreamExt};
use pin_project::pin_project;

use tc_error::TCResult;

// Based on: https://github.com/rust-lang/futures-rs/blob/master/futures-util/src/stream/select.rs
#[pin_project]
pub struct SparseCombine<T, S, O> {
    #[pin]
    left: Fuse<S>,
    #[pin]
    right: Fuse<S>,

    offset: O,

    pending_left: Option<(u64, T)>,
    pending_right: Option<(u64, T)>,
}

impl<'a, T, S: Stream<Item = TCResult<T>>, O: Fn(&T) -> u64> SparseCombine<T, S, O> {
    pub fn new(left: S, right: S, offset: O) -> Self {
        Self {
            left: left.fuse(),
            right: right.fuse(),
            offset,
            pending_left: None,
            pending_right: None,
        }
    }

    fn poll_inner(
        offset: &O,
        stream: Pin<&mut Fuse<S>>,
        pending: &mut Option<(u64, T)>,
        cxt: &mut task::Context,
    ) -> TCResult<bool> {
        match stream.poll_next(cxt) {
            Poll::Pending => Ok(false),
            Poll::Ready(Some(Ok(value))) => {
                let offset = offset(&value);
                *pending = Some((offset, value));
                Ok(false)
            }
            Poll::Ready(Some(Err(cause))) => Err(cause),
            Poll::Ready(None) => Ok(true),
        }
    }

    fn swap_value(pending: &mut Option<(u64, T)>) -> T {
        assert!(pending.is_some());

        let mut value: Option<(u64, T)> = None;
        mem::swap(pending, &mut value);
        let (_, value) = value.unwrap();
        value
    }
}

impl<T, S: Stream<Item = TCResult<T>>, O: Fn(&T) -> u64> Stream for SparseCombine<T, S, O> {
    type Item = TCResult<(Option<T>, Option<T>)>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut task::Context) -> Poll<Option<Self::Item>> {
        let this = self.project();

        let left_done = if this.left.is_done() {
            true
        } else if this.pending_left.is_none() {
            match Self::poll_inner(this.offset, this.left, this.pending_left, cxt) {
                Err(cause) => return Poll::Ready(Some(Err(cause))),
                Ok(done) => done,
            }
        } else {
            false
        };

        let right_done = if this.right.is_done() {
            true
        } else if this.pending_right.is_none() {
            match Self::poll_inner(this.offset, this.right, this.pending_right, cxt) {
                Err(cause) => return Poll::Ready(Some(Err(cause))),
                Ok(done) => done,
            }
        } else {
            false
        };

        if this.pending_left.is_some() && this.pending_right.is_some() {
            let (l_offset, _) = this.pending_left.as_ref().unwrap();
            let (r_offset, _) = this.pending_right.as_ref().unwrap();

            match l_offset.cmp(r_offset) {
                Ordering::Equal => {
                    let l_value = Self::swap_value(this.pending_left);
                    let r_value = Self::swap_value(this.pending_right);
                    Poll::Ready(Some(Ok((Some(l_value), Some(r_value)))))
                }
                Ordering::Less => {
                    let l_value = Self::swap_value(this.pending_left);
                    Poll::Ready(Some(Ok((Some(l_value), None))))
                }
                Ordering::Greater => {
                    let r_value = Self::swap_value(this.pending_right);
                    Poll::Ready(Some(Ok((None, Some(r_value)))))
                }
            }
        } else if right_done && this.pending_left.is_some() {
            let l_value = Self::swap_value(this.pending_left);
            Poll::Ready(Some(Ok((Some(l_value), None))))
        } else if left_done && this.pending_right.is_some() {
            let r_value = Self::swap_value(this.pending_right);
            Poll::Ready(Some(Ok((None, Some(r_value)))))
        } else if left_done && right_done {
            Poll::Ready(None)
        } else {
            Poll::Pending
        }
    }
}

#[inline]
pub fn coord_to_offset(coord: &[u64], coord_bounds: &[u64]) -> u64 {
    coord_bounds
        .iter()
        .zip(coord.iter())
        .map(|(d, x)| d * x)
        .sum()
}
