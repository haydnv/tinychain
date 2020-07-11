use std::collections::VecDeque;
use std::iter;
use std::ops;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::stream::Stream;
use itertools::MultiProduct;

use crate::value::class::NumberType;
use crate::value::{Number, TCResult};

use super::chunk::ChunkData;

pub type Coords = MultiProduct<AxisIter>;

#[derive(Clone)]
pub enum AxisIter {
    One(std::iter::Once<u64>),
    Each(Vec<u64>, usize),
    Step(iter::StepBy<ops::Range<u64>>),
}

impl Iterator for AxisIter {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        use AxisIter::*;
        match self {
            One(iter) => iter.next(),
            Each(v, at) => {
                if at == &v.len() {
                    None
                } else {
                    Some(v[*at])
                }
            }
            Step(iter) => iter.next(),
        }
    }
}

pub struct ValueChunkStream<T: Stream<Item = TCResult<Number>>> {
    source: Pin<Box<T>>,
    dtype: NumberType,
    chunk_len: usize,
    buffer: Vec<Number>,
    valid: bool,
}

impl<T: Stream<Item = TCResult<Number>>> ValueChunkStream<T> {
    pub fn new(source: T, dtype: NumberType, chunk_len: usize) -> ValueChunkStream<T> {
        let buffer = Vec::with_capacity(chunk_len);

        ValueChunkStream {
            source: Box::pin(source),
            dtype,
            chunk_len,
            buffer,
            valid: true,
        }
    }
}

impl<T: Stream<Item = TCResult<Number>>> Stream for ValueChunkStream<T> {
    type Item = TCResult<ChunkData>;

    fn poll_next(mut self: Pin<&mut Self>, cxt: &mut Context) -> Poll<Option<Self::Item>> {
        if !self.valid {
            Poll::Ready(None)
        } else if self.buffer.len() == self.chunk_len {
            match ChunkData::try_from_values(self.buffer.drain(..).collect(), self.dtype) {
                Ok(chunk) => Poll::Ready(Some(Ok(chunk))),
                Err(cause) => {
                    self.valid = false;
                    Poll::Ready(Some(Err(cause)))
                }
            }
        } else {
            match Pin::as_mut(&mut self.source).poll_next(cxt) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(Some(Err(cause))) => {
                    self.valid = false;
                    Poll::Ready(Some(Err(cause)))
                }
                Poll::Ready(Some(Ok(value))) => {
                    self.buffer.push(value);
                    Poll::Pending
                }
                Poll::Ready(None) if self.buffer.is_empty() => Poll::Ready(None),
                Poll::Ready(None) => {
                    match ChunkData::try_from_values(self.buffer.drain(..).collect(), self.dtype) {
                        Ok(chunk) => Poll::Ready(Some(Ok(chunk))),
                        Err(cause) => {
                            self.valid = false;
                            Poll::Ready(Some(Err(cause)))
                        }
                    }
                }
            }
        }
    }
}

pub struct ValueStream<T: Stream<Item = TCResult<ChunkData>>> {
    source: Pin<Box<T>>,
    buffer: VecDeque<Number>,
    valid: bool,
}

impl<T: Stream<Item = TCResult<ChunkData>>> ValueStream<T> {
    pub fn new(source: T) -> ValueStream<T> {
        ValueStream {
            source: Box::pin(source),
            buffer: VecDeque::new(),
            valid: true,
        }
    }
}

impl<T: Stream<Item = TCResult<ChunkData>>> Stream for ValueStream<T> {
    type Item = TCResult<Number>;

    fn poll_next(mut self: Pin<&mut Self>, cxt: &mut Context) -> Poll<Option<Self::Item>> {
        if !self.valid {
            Poll::Ready(None)
        } else if let Some(value) = self.buffer.pop_front() {
            Poll::Ready(Some(Ok(value)))
        } else {
            match Pin::as_mut(&mut self.source).poll_next(cxt) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(Some(Ok(chunk))) => {
                    let buffered: Vec<Number> = chunk.into();
                    self.buffer.extend(buffered);
                    Poll::Ready(self.buffer.pop_front().map(|v| Ok(v)))
                }
                Poll::Ready(Some(Err(cause))) => {
                    self.valid = false;
                    Poll::Ready(Some(Err(cause)))
                }
                Poll::Ready(None) => Poll::Ready(None),
            }
        }
    }
}
