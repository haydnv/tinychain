use std::pin::Pin;
use std::task::{Context, Poll};

use async_trait::async_trait;
use futures::stream::{Stream, StreamExt};
use safecast::{CastFrom, CastInto};

use tc_error::*;
use tc_value::{Number, NumberInstance, UInt, Value};
use tcgeneric::TCBoxTryStream;

use crate::state::State;
use crate::txn::Txn;

use super::source::Source;
use super::TCStream;

#[derive(Clone)]
pub struct Range {
    start: Number,
    stop: Number,
    step: Number,
}

impl Range {
    pub fn new(start: Number, stop: Number, step: Number) -> Self {
        Self { start, stop, step }
    }
}

#[async_trait]
impl Source for Range {
    async fn into_stream(self, _txn: Txn) -> TCResult<TCBoxTryStream<'static, State>> {
        let range = RangeStream::new(self.start, self.stop, self.step)
            .map(Value::Number)
            .map(State::from)
            .map(Ok);

        let range: TCBoxTryStream<_> = Box::pin(range);
        Ok(range)
    }
}

impl From<Range> for TCStream {
    fn from(range: Range) -> Self {
        TCStream::Range(range)
    }
}

struct RangeStream {
    current: Number,
    step: Number,
    stop: Number,
}

impl RangeStream {
    fn new(start: Number, stop: Number, step: Number) -> Self {
        Self {
            current: start,
            stop,
            step: if start < stop {
                step.abs()
            } else {
                Number::from(0) - step.abs()
            },
        }
    }
}

impl Stream for RangeStream {
    type Item = Number;

    fn poll_next(mut self: Pin<&mut Self>, _cxt: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.current >= self.stop {
            Poll::Ready(None)
        } else {
            let next = self.current;
            self.current = next + self.step;
            Poll::Ready(Some(next))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = (self.stop - self.current) / self.step;
        match size {
            Number::Bool(_) => (0, Some(1)),
            Number::Complex(size) => {
                let size = UInt::cast_from(size).cast_into();
                (size, Some(size + 1))
            }
            Number::Float(size) => {
                let size = UInt::cast_from(size).cast_into();
                (size, Some(size + 1))
            }
            Number::Int(size) => {
                let size = UInt::cast_from(size).cast_into();
                (size, Some(size))
            }
            Number::UInt(size) => {
                let size = UInt::cast_from(size).cast_into();
                (size, Some(size))
            }
        }
    }
}
