use std::cmp::Ordering;
use std::pin::Pin;
use std::task::{ready, Context, Poll};
use std::{fmt, mem};

use futures::stream::{Fuse, Stream, StreamExt, TryStream};
use ha_ndarray::{shape, ArrayBuf, Buffer, CType};
use pin_project::pin_project;
use smallvec::SmallVec;

use tc_error::*;
use tc_transact::lock::PermitRead;

use crate::tensor::{Axes, Coord, Range, IDEAL_BLOCK_SIZE};

#[pin_project]
pub struct BlockCoords<S, T> {
    #[pin]
    source: Fuse<S>,
    pending_coords: Vec<u64>,
    pending_values: Vec<T>,
    ndim: usize,
}

impl<S, T> BlockCoords<S, T>
where
    S: Stream<Item = TCResult<(Coord, T)>>,
{
    pub fn new(source: S, ndim: usize) -> Self {
        Self {
            source: source.fuse(),
            pending_coords: Vec::with_capacity(IDEAL_BLOCK_SIZE * ndim),
            pending_values: Vec::with_capacity(IDEAL_BLOCK_SIZE),
            ndim,
        }
    }
}

impl<S, T> BlockCoords<S, T>
where
    T: CType,
{
    fn block_cutoff(
        pending_coords: &mut Vec<u64>,
        pending_values: &mut Vec<T>,
        ndim: usize,
    ) -> TCResult<(ArrayBuf<u64, Buffer<u64>>, ArrayBuf<T, Buffer<T>>)> {
        let num_coords = pending_values.len();

        debug_assert_eq!(pending_coords.len() % ndim, 0);

        let buf = Buffer::from_slice(&pending_values)?;
        let values = ArrayBuf::new(buf, shape![num_coords])?;
        pending_values.clear();

        let buf = Buffer::from_slice(&pending_coords)?;
        let coords = ArrayBuf::new(buf, shape![pending_coords.len() / ndim, ndim])?;
        pending_coords.clear();

        Ok((coords, values))
    }
}

impl<S, T> Stream for BlockCoords<S, T>
where
    S: Stream<Item = TCResult<(Coord, T)>> + Unpin,
    T: CType,
{
    type Item = TCResult<(ArrayBuf<u64, Buffer<u64>>, ArrayBuf<T, Buffer<T>>)>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context) -> Poll<Option<Self::Item>> {
        let ndim = self.ndim;
        let mut this = self.project();

        Poll::Ready(loop {
            debug_assert_eq!(this.pending_values.len() * ndim, this.pending_coords.len());

            match ready!(this.source.as_mut().poll_next(cxt)) {
                Some(Ok((coord, value))) => {
                    debug_assert_eq!(coord.len(), *this.ndim);

                    this.pending_coords.extend(coord);
                    this.pending_values.push(value);

                    if this.pending_values.len() == IDEAL_BLOCK_SIZE {
                        break Some(Self::block_cutoff(
                            this.pending_coords,
                            this.pending_values,
                            ndim,
                        ));
                    }
                }
                None if !this.pending_values.is_empty() => {
                    break Some(Self::block_cutoff(
                        this.pending_coords,
                        this.pending_values,
                        ndim,
                    ));
                }
                None => break None,
                Some(Err(cause)) => break Some(Err(cause)),
            }
        })
    }
}

#[pin_project]
pub struct BlockOffsetsDual<S, T> {
    #[pin]
    source: Fuse<S>,
    pending_offsets: Vec<u64>,
    pending_left: Vec<T>,
    pending_right: Vec<T>,
}

impl<S, T> BlockOffsetsDual<S, T>
where
    S: Stream<Item = TCResult<(u64, (T, T))>>,
{
    pub fn new(source: S) -> Self {
        Self {
            source: source.fuse(),
            pending_offsets: Vec::with_capacity(IDEAL_BLOCK_SIZE),
            pending_left: Vec::with_capacity(IDEAL_BLOCK_SIZE),
            pending_right: Vec::with_capacity(IDEAL_BLOCK_SIZE),
        }
    }
}

impl<S, T> BlockOffsetsDual<S, T>
where
    T: CType,
{
    fn block_cutoff(
        pending_offsets: &mut Vec<u64>,
        pending_left: &mut Vec<T>,
        pending_right: &mut Vec<T>,
    ) -> TCResult<(
        ArrayBuf<u64, Buffer<u64>>,
        (ArrayBuf<T, Buffer<T>>, ArrayBuf<T, Buffer<T>>),
    )> {
        debug_assert_eq!(pending_offsets.len(), pending_left.len());
        debug_assert_eq!(pending_left.len(), pending_right.len());

        let num_offsets = pending_left.len();

        let buf = Buffer::from_slice(&pending_left)?;
        let left = ArrayBuf::new(buf, shape![num_offsets])?;
        pending_left.clear();

        let buf = Buffer::from_slice(&pending_right)?;
        let right = ArrayBuf::new(buf, shape![num_offsets])?;
        pending_right.clear();

        let buf = Buffer::from_slice(&pending_offsets)?;
        let coords = ArrayBuf::new(buf, shape![num_offsets])?;
        pending_offsets.clear();

        Ok((coords, (left, right)))
    }
}

impl<S, T> Stream for BlockOffsetsDual<S, T>
where
    S: Stream<Item = TCResult<(u64, (T, T))>> + Unpin,
    T: CType,
{
    type Item = TCResult<(
        ArrayBuf<u64, Buffer<u64>>,
        (ArrayBuf<T, Buffer<T>>, ArrayBuf<T, Buffer<T>>),
    )>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        Poll::Ready(loop {
            debug_assert_eq!(this.pending_left.len(), this.pending_right.len());
            debug_assert_eq!(this.pending_left.len(), this.pending_offsets.len());

            match ready!(this.source.as_mut().poll_next(cxt)) {
                Some(Ok((offset, (left, right)))) => {
                    this.pending_offsets.push(offset);
                    this.pending_left.push(left);
                    this.pending_right.push(right);

                    if this.pending_offsets.len() == IDEAL_BLOCK_SIZE {
                        break Some(Self::block_cutoff(
                            this.pending_offsets,
                            this.pending_left,
                            this.pending_right,
                        ));
                    }
                }
                None if !this.pending_offsets.is_empty() => {
                    break Some(Self::block_cutoff(
                        this.pending_offsets,
                        this.pending_left,
                        this.pending_right,
                    ));
                }
                None => break None,
                Some(Err(cause)) => break Some(Err(cause)),
            }
        })
    }
}

#[pin_project]
pub struct BlockOffsets<S, T> {
    #[pin]
    source: Fuse<S>,
    pending_offsets: Vec<u64>,
    pending_values: Vec<T>,
}

impl<S, T> BlockOffsets<S, T>
where
    S: Stream<Item = TCResult<(u64, T)>>,
{
    pub fn new(source: S) -> Self {
        Self {
            source: source.fuse(),
            pending_offsets: Vec::with_capacity(IDEAL_BLOCK_SIZE),
            pending_values: Vec::with_capacity(IDEAL_BLOCK_SIZE),
        }
    }
}

impl<S, T> BlockOffsets<S, T>
where
    T: CType,
{
    fn block_cutoff(
        pending_offsets: &mut Vec<u64>,
        pending_values: &mut Vec<T>,
    ) -> TCResult<(ArrayBuf<u64, Buffer<u64>>, ArrayBuf<T, Buffer<T>>)> {
        debug_assert_eq!(pending_offsets.len(), pending_values.len());

        let buf = Buffer::from_slice(&pending_values)?;
        let values = ArrayBuf::new(buf, shape![pending_values.len()])?;
        pending_values.clear();

        let buf = Buffer::from_slice(&pending_offsets)?;
        let offsets = ArrayBuf::new(buf, shape![pending_offsets.len()])?;
        pending_offsets.clear();

        Ok((offsets, values))
    }
}

impl<S, T> Stream for BlockOffsets<S, T>
where
    S: Stream<Item = TCResult<(u64, T)>> + Unpin,
    T: CType,
{
    type Item = TCResult<(ArrayBuf<u64, Buffer<u64>>, ArrayBuf<T, Buffer<T>>)>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        Poll::Ready(loop {
            match ready!(this.source.as_mut().poll_next(cxt)) {
                Some(Ok((offset, value))) => {
                    this.pending_offsets.push(offset);
                    this.pending_values.push(value);

                    if this.pending_values.len() == IDEAL_BLOCK_SIZE {
                        break Some(Self::block_cutoff(
                            this.pending_offsets,
                            this.pending_values,
                        ));
                    }
                }
                None if !this.pending_values.is_empty() => {
                    break Some(Self::block_cutoff(
                        this.pending_offsets,
                        this.pending_values,
                    ));
                }
                None => break None,
                Some(Err(cause)) => break Some(Err(cause)),
            }
        })
    }
}

#[pin_project]
pub struct Elements<S> {
    permit: SmallVec<[PermitRead<Range>; 16]>,

    #[pin]
    elements: Fuse<S>,
}

impl<S> Elements<S>
where
    S: Stream,
{
    pub fn new(permit: SmallVec<[PermitRead<Range>; 16]>, elements: S) -> Self {
        Self {
            permit,
            elements: elements.fuse(),
        }
    }
}

impl<S, T> Stream for Elements<S>
where
    S: Stream<Item = TCResult<(Coord, T)>>,
{
    type Item = TCResult<(Coord, T)>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        Poll::Ready(match ready!(this.elements.as_mut().try_poll_next(cxt)) {
            Some(Ok((coord, n))) => Some(Ok((coord, n))),
            Some(Err(cause)) => Some(Err(cause)),
            None => None,
        })
    }
}

#[pin_project]
pub struct Select<Cond, Then, OrElse, T> {
    #[pin]
    cond: Fuse<Cond>,
    #[pin]
    then: Fuse<Then>,
    #[pin]
    or_else: Fuse<OrElse>,

    pending_cond: Option<(u64, u8)>,
    pending_then: Option<(u64, T)>,
    pending_else: Option<(u64, T)>,
}

impl<Cond, Then, OrElse, T> Select<Cond, Then, OrElse, T>
where
    Cond: Stream,
    Then: Stream,
    OrElse: Stream,
{
    pub fn new(cond: Cond, then: Then, or_else: OrElse) -> Self {
        Self {
            cond: cond.fuse(),
            then: then.fuse(),
            or_else: or_else.fuse(),
            pending_cond: None,
            pending_then: None,
            pending_else: None,
        }
    }
}

// Based on: https://github.com/rust-lang/futures-rs/blob/master/futures-util/src/stream/select.rs
impl<Cond, Then, OrElse, T> Stream for Select<Cond, Then, OrElse, T>
where
    Cond: Stream<Item = TCResult<(u64, u8)>>,
    Then: Stream<Item = TCResult<(u64, T)>>,
    OrElse: Stream<Item = TCResult<(u64, T)>>,
    T: CType + fmt::Debug,
{
    type Item = TCResult<(u64, T)>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        Poll::Ready(loop {
            if !this.cond.is_done() && this.pending_cond.is_none() {
                match ready!(this.cond.as_mut().try_poll_next(cxt)) {
                    Some(Ok(value)) => *this.pending_cond = Some(value),
                    Some(Err(cause)) => break Some(Err(cause)),
                    None => {}
                }
            }

            if !this.then.is_done() && this.pending_then.is_none() {
                match ready!(this.then.as_mut().try_poll_next(cxt)) {
                    Some(Ok(value)) => *this.pending_then = Some(value),
                    Some(Err(cause)) => break Some(Err(cause)),
                    None => {}
                }
            }

            if !this.or_else.is_done() && this.pending_else.is_none() {
                match ready!(this.or_else.as_mut().try_poll_next(cxt)) {
                    Some(Ok(value)) => *this.pending_else = Some(value),
                    Some(Err(cause)) => break Some(Err(cause)),
                    None => {}
                }
            }

            let cond = this.pending_cond.as_ref().map(|(offset, _)| offset);
            let then = this.pending_then.as_ref().map(|(offset, _)| *offset);
            let or_else = this.pending_else.as_ref().map(|(offset, _)| *offset);

            match (cond, then, or_else) {
                (Some(offset), _then, Some(else_offset)) if else_offset < *offset => {
                    // the cond stream skipped over a filled false-value
                    break this.pending_else.take().map(Ok);
                }
                (Some(offset), Some(then_offset), _else) => match offset.cmp(&then_offset) {
                    Ordering::Less => {
                        // consume this element in the cond stream
                        *this.pending_cond = None
                    }
                    Ordering::Equal => {
                        *this.pending_cond = None; // consume this element in the cond stream
                        break this.pending_then.take().map(Ok); // consume this true-value
                    }
                    Ordering::Greater => {
                        *this.pending_then = None; // consume a skipped true-value
                    }
                },
                (Some(offset), None, Some(else_offset)) => match offset.cmp(&else_offset) {
                    Ordering::Less => {
                        *this.pending_cond = None; // consume this element in the cond stream
                    }
                    Ordering::Equal => {
                        *this.pending_cond = None; // consume this element in the cond stream
                        *this.pending_else = None; // consume this false-value
                    }
                    Ordering::Greater => {
                        // consume and return this false-value
                        break this.pending_else.take().map(Ok);
                    }
                },
                (None, _then, _else) => {
                    // consume and return the next false-value, if any
                    break this.pending_else.take().map(Ok);
                }
                (_cond, None, None) => {
                    // consume this element in the cond stream in case poll is called again
                    *this.pending_cond = None;

                    // there are no more values to return
                    break None;
                }
            }
        })
    }
}

#[pin_project]
pub struct TryDiff<L, R, T> {
    #[pin]
    left: Fuse<L>,
    #[pin]
    right: Fuse<R>,

    pending_left: Option<(u64, T)>,
    pending_right: Option<(u64, T)>,
}

impl<L, R, T> TryDiff<L, R, T>
where
    L: Stream,
    R: Stream,
{
    pub fn new(left: L, right: R) -> Self {
        Self {
            left: left.fuse(),
            right: right.fuse(),
            pending_left: None,
            pending_right: None,
        }
    }
}

// Based on: https://github.com/rust-lang/futures-rs/blob/master/futures-util/src/stream/select.rs
impl<L, R, T> Stream for TryDiff<L, R, T>
where
    L: Stream<Item = TCResult<(u64, T)>>,
    R: Stream<Item = TCResult<(u64, T)>>,
    T: CType + fmt::Debug,
{
    type Item = TCResult<(u64, T)>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        Poll::Ready(loop {
            let left_done = if this.left.is_done() {
                true
            } else if this.pending_left.is_none() {
                match ready!(this.left.as_mut().try_poll_next(cxt)) {
                    Some(Ok(value)) => {
                        *this.pending_left = Some(value);
                        false
                    }
                    Some(Err(cause)) => break Some(Err(cause)),
                    None => true,
                }
            } else {
                false
            };

            let right_done = if this.right.is_done() {
                true
            } else if this.pending_right.is_none() {
                match ready!(this.right.as_mut().try_poll_next(cxt)) {
                    Some(Ok(value)) => {
                        *this.pending_right = Some(value);
                        false
                    }
                    Some(Err(cause)) => break Some(Err(cause)),
                    None => true,
                }
            } else {
                false
            };

            if this.pending_left.is_some() && this.pending_right.is_some() {
                let (l_offset, _value) = this.pending_left.as_ref().unwrap();
                let (r_offset, zero) = this.pending_right.as_ref().unwrap();
                debug_assert_eq!(*zero, T::ZERO);

                match l_offset.cmp(r_offset) {
                    Ordering::Equal => {
                        // this value has been zero'd out, so drop it
                        this.pending_left.take();
                        this.pending_right.take();
                    }
                    Ordering::Less => {
                        // this value is not present in the right stream, so return it
                        break this.pending_left.take().map(Ok);
                    }
                    Ordering::Greater => {
                        // this value could be present in the right stream--wait and see
                        this.pending_right.take();
                    }
                }
            } else if right_done && this.pending_left.is_some() {
                break this.pending_left.take().map(Ok);
            } else if left_done {
                break None;
            }
        })
    }
}

#[pin_project]
pub struct FilledAt<S> {
    #[pin]
    source: S,

    pending: Option<SmallVec<[u64; 8]>>,
    axes: Axes,
    ndim: usize,
}

impl<S> FilledAt<S> {
    pub fn new(source: S, axes: Axes, ndim: usize) -> Self {
        debug_assert!(!axes.is_empty());
        debug_assert!(!axes.iter().copied().any(|x| x >= ndim));

        Self {
            source,
            pending: None,
            axes,
            ndim,
        }
    }
}

impl<T, S: Stream<Item = TCResult<(Coord, T)>>> Stream for FilledAt<S> {
    type Item = TCResult<Coord>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        Poll::Ready(loop {
            match ready!(this.source.as_mut().poll_next(cxt)) {
                Some(Ok((coord, _))) => match this.pending.as_mut() {
                    None => {
                        let filled = this.axes.iter().copied().map(|x| coord[x]).collect();
                        *this.pending = Some(filled);
                    }
                    Some(pending) => {
                        if this
                            .axes
                            .iter()
                            .copied()
                            .map(|x| coord[x])
                            .zip(pending.iter().copied())
                            .any(|(l, r)| l != r)
                        {
                            let mut filled =
                                Some(this.axes.iter().copied().map(|x| coord[x]).collect());

                            mem::swap(&mut *this.pending, &mut filled);
                            break filled.map(Ok);
                        }
                    }
                },
                None => break this.pending.take().map(Ok),
                Some(Err(cause)) => break Some(Err(cause)),
            }
        })
    }
}

// Based on: https://github.com/rust-lang/futures-rs/blob/master/futures-util/src/stream/select.rs
#[pin_project]
pub struct InnerJoin<L, R, T> {
    #[pin]
    left: Fuse<L>,
    #[pin]
    right: Fuse<R>,

    pending_left: Option<(u64, T)>,
    pending_right: Option<(u64, T)>,
}

impl<L, R, T> InnerJoin<L, R, T>
where
    L: Stream<Item = TCResult<(u64, T)>>,
    R: Stream<Item = TCResult<(u64, T)>>,
{
    pub fn new(left: L, right: R) -> Self {
        Self {
            left: left.fuse(),
            right: right.fuse(),
            pending_left: None,
            pending_right: None,
        }
    }
}

impl<L, R, T> Stream for InnerJoin<L, R, T>
where
    L: Stream<Item = TCResult<(u64, T)>>,
    R: Stream<Item = TCResult<(u64, T)>>,
    T: fmt::Debug,
{
    type Item = TCResult<(u64, (T, T))>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        Poll::Ready(loop {
            let left_done = if this.left.is_done() {
                true
            } else if this.pending_left.is_none() {
                match ready!(this.left.as_mut().poll_next(cxt)) {
                    Some(Err(cause)) => break Some(Err(cause)),
                    Some(Ok(pending_left)) => {
                        *this.pending_left = Some(pending_left);
                        false
                    }
                    None => true,
                }
            } else {
                false
            };

            let right_done = if this.right.is_done() {
                true
            } else if this.pending_right.is_none() {
                match ready!(this.right.as_mut().poll_next(cxt)) {
                    Some(Err(cause)) => break Some(Err(cause)),
                    Some(Ok(pending_right)) => {
                        *this.pending_right = Some(pending_right);
                        false
                    }
                    None => true,
                }
            } else {
                false
            };

            // TODO: is there a way to structure this without calling unwrap so many times?
            if this.pending_left.is_some() && this.pending_right.is_some() {
                let l_offset = this.pending_left.as_ref().unwrap().0;
                let r_offset = this.pending_right.as_ref().unwrap().0;

                log::trace!("inner join {:?} and {:?}?", l_offset, r_offset);

                match l_offset.cmp(&r_offset) {
                    Ordering::Equal => {
                        let (l_offset, l_value) = this.pending_left.take().unwrap();
                        let (_r_offset, r_value) = this.pending_right.take().unwrap();
                        break Some(Ok((l_offset, (l_value, r_value))));
                    }
                    Ordering::Less => {
                        this.pending_left.take();
                    }
                    Ordering::Greater => {
                        this.pending_right.take();
                    }
                }
            } else if left_done || right_done {
                break None;
            }
        })
    }
}

// Based on: https://github.com/rust-lang/futures-rs/blob/master/futures-util/src/stream/select.rs
#[pin_project]
pub struct TryMerge<L, R, T> {
    #[pin]
    left: Fuse<L>,
    #[pin]
    right: Fuse<R>,

    pending_left: Option<(u64, T)>,
    pending_right: Option<(u64, T)>,
}

impl<L, R, T> TryMerge<L, R, T>
where
    L: Stream<Item = TCResult<(u64, T)>>,
    R: Stream<Item = TCResult<(u64, T)>>,
{
    pub fn new(left: L, right: R) -> Self {
        Self {
            left: left.fuse(),
            right: right.fuse(),
            pending_left: None,
            pending_right: None,
        }
    }
}

impl<L, R, T> Stream for TryMerge<L, R, T>
where
    Fuse<L>: TryStream<Ok = (u64, T), Error = TCError> + Unpin,
    Fuse<R>: TryStream<Ok = (u64, T), Error = TCError> + Unpin,
    T: CType + fmt::Debug,
{
    type Item = TCResult<(u64, T)>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context) -> Poll<Option<Self::Item>> {
        let this = self.project();

        let left_done = if this.left.is_done() {
            true
        } else if this.pending_left.is_none() {
            match ready!(this.left.try_poll_next(cxt)) {
                Some(Ok(value)) => {
                    *this.pending_left = Some(value);
                    false
                }
                Some(Err(cause)) => return Poll::Ready(Some(Err(cause))),
                None => true,
            }
        } else {
            false
        };

        let right_done = if this.right.is_done() {
            true
        } else if this.pending_right.is_none() {
            match ready!(this.right.try_poll_next(cxt)) {
                Some(Ok(value)) => {
                    *this.pending_right = Some(value);
                    false
                }
                Some(Err(cause)) => return Poll::Ready(Some(Err(cause))),
                None => true,
            }
        } else {
            false
        };

        let value = if this.pending_left.is_some() && this.pending_right.is_some() {
            let (l_offset, l_value) = this.pending_left.as_ref().unwrap();
            let (r_offset, r_value) = this.pending_right.as_ref().unwrap();

            debug_assert_ne!(*l_value, T::ZERO);
            debug_assert_ne!(*r_value, T::ZERO);

            match l_offset.cmp(r_offset) {
                Ordering::Equal => {
                    this.pending_left.take();
                    this.pending_right.take()
                }
                Ordering::Less => this.pending_left.take(),
                Ordering::Greater => this.pending_right.take(),
            }
        } else if right_done && this.pending_left.is_some() {
            this.pending_left.take()
        } else if left_done && this.pending_right.is_some() {
            this.pending_right.take()
        } else if left_done && right_done {
            None
        } else {
            unreachable!("both streams to merge are still pending")
        };

        Poll::Ready(value.map(Ok))
    }
}

// Based on: https://github.com/rust-lang/futures-rs/blob/master/futures-util/src/stream/select.rs
#[pin_project]
pub struct OuterJoin<L, R, T> {
    #[pin]
    left: Fuse<L>,
    #[pin]
    right: Fuse<R>,

    zero: T,
    pending_left: Option<(u64, T)>,
    pending_right: Option<(u64, T)>,
}

impl<L, R, T> OuterJoin<L, R, T>
where
    L: Stream<Item = TCResult<(u64, T)>>,
    R: Stream<Item = TCResult<(u64, T)>>,
{
    pub fn new(left: L, right: R, zero: T) -> Self {
        Self {
            left: left.fuse(),
            right: right.fuse(),
            zero,
            pending_left: None,
            pending_right: None,
        }
    }
}

impl<L, R, T> Stream for OuterJoin<L, R, T>
where
    L: Stream<Item = TCResult<(u64, T)>>,
    R: Stream<Item = TCResult<(u64, T)>>,
    T: Copy + PartialEq + fmt::Debug,
{
    type Item = TCResult<(u64, (T, T))>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        Poll::Ready(loop {
            let left_done = if this.left.is_done() {
                true
            } else if this.pending_left.is_none() {
                match ready!(this.left.as_mut().poll_next(cxt)) {
                    Some(Err(cause)) => break Some(Err(cause)),
                    Some(Ok(pending_left)) => {
                        *this.pending_left = Some(pending_left);
                        false
                    }
                    None => true,
                }
            } else {
                false
            };

            let right_done = if this.right.is_done() {
                true
            } else if this.pending_right.is_none() {
                match ready!(this.right.as_mut().poll_next(cxt)) {
                    Some(Err(cause)) => break Some(Err(cause)),
                    Some(Ok(pending_right)) => {
                        *this.pending_right = Some(pending_right);
                        false
                    }
                    None => true,
                }
            } else {
                false
            };

            log::trace!(
                "sparse outer join state: ({:?}, {:?})",
                this.pending_left,
                this.pending_right
            );

            if this.pending_left.is_some() && this.pending_right.is_some() {
                let l_offset = this.pending_left.as_ref().unwrap().0;
                let r_offset = this.pending_right.as_ref().unwrap().0;

                break match l_offset.cmp(&r_offset) {
                    Ordering::Equal => {
                        let (_, l_value) = this.pending_left.take().unwrap();
                        let (_, r_value) = this.pending_right.take().unwrap();
                        Some(Ok((l_offset, (l_value, r_value))))
                    }
                    Ordering::Less => {
                        let (offset, l_value) = this.pending_left.take().unwrap();
                        Some(Ok((offset, (l_value, *this.zero))))
                    }
                    Ordering::Greater => {
                        let (offset, r_value) = this.pending_right.take().unwrap();
                        Some(Ok((offset, (*this.zero, r_value))))
                    }
                };
            } else if right_done && this.pending_left.is_some() {
                let (offset, l_value) = this.pending_left.take().unwrap();
                break Some(Ok((offset, (l_value, *this.zero))));
            } else if left_done && this.pending_right.is_some() {
                let (offset, r_value) = this.pending_right.take().unwrap();
                break Some(Ok((offset, (*this.zero, r_value))));
            } else if left_done && right_done {
                break None;
            }
        })
    }
}
