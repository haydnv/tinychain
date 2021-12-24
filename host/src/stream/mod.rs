//! A stream generator such as a `Collection` or a mapping or aggregation of its items

use std::convert::TryInto;
use std::pin::Pin;
use std::task::{Context, Poll};

use async_trait::async_trait;
use destream::en;
use futures::future::{self, TryFutureExt};
use futures::stream::{Stream, StreamExt, TryStreamExt};
use log::debug;
use safecast::{CastFrom, CastInto, TryCastFrom};

use tc_error::*;
use tc_transact::IntoView;
use tc_value::{Number, UInt};
use tcgeneric::{Id, TCBoxTryFuture, TCBoxTryStream};

use crate::closure::Closure;
use crate::fs;
use crate::state::{State, StateView};
use crate::txn::Txn;
use crate::value::Value;

use source::*;

mod group;
mod source;

/// A stream generator such as a `Collection` or a mapping or aggregation of its items
#[derive(Clone)]
pub enum TCStream {
    Aggregate(Box<Aggregate>),
    Collection(Collection),
    Filter(Box<Filter>),
    Flatten(Box<TCStream>),
    Map(Box<Map>),
    Range(Number, Number, Number),
}

impl TCStream {
    /// Group equal sequential items in this stream.
    ///
    /// For example, aggregating the stream `['b', 'b', 'a', 'a', 'b']`
    /// will produce `['b', 'a', 'b']`.
    pub fn aggregate(self) -> Self {
        Aggregate::new(self).into()
    }

    /// Return a new stream with only the elements in this stream which match the given `filter`.
    pub fn filter(self, filter: Closure) -> Self {
        Filter::new(self, filter).into()
    }

    /// Flatten a stream of streams into a stream of `State`s.
    pub fn flatten(self) -> Self {
        Self::Flatten(Box::new(self))
    }

    /// Fold this stream with the given initial `State` and `Closure`.
    ///
    /// For example, folding `[1, 2, 3]` with `0` and `Number::add` will produce `6`.
    pub async fn fold(
        self,
        txn: Txn,
        item_name: Id,
        mut state: tcgeneric::Map<State>,
        op: Closure,
    ) -> TCResult<State> {
        let mut source = self.into_stream(txn.clone()).await?;

        while let Some(item) = source.try_next().await? {
            let mut args = state.clone();
            args.insert(item_name.clone(), item);
            let result = op.clone().call(&txn, args.into()).await?;
            state = result.try_into()?;
        }

        Ok(State::Map(state))
    }

    /// Execute the given [`Closure`] with each item in the stream as its `args`.
    pub async fn for_each(self, txn: &Txn, op: Closure) -> TCResult<()> {
        debug!("Stream::for_each {}", op);

        let stream = self.into_stream(txn.clone()).await?;

        stream
            .map_ok(move |args| {
                debug!("Stream::for_each calling op with args {}", args);
                op.clone().call(&txn, args)
            })
            .try_buffer_unordered(num_cpus::get())
            .try_fold((), |(), _none| future::ready(Ok(())))
            .await
    }

    /// Return a `TCStream` produced by calling the given [`Closure`] on each item in this stream.
    pub fn map(self, op: Closure) -> Self {
        Map::new(self, op).into()
    }

    /// Return a Rust `Stream` of the items in this `TCStream`.
    pub fn into_stream<'a>(self, txn: Txn) -> TCBoxTryFuture<'a, TCBoxTryStream<'static, State>> {
        Box::pin(async move {
            match self {
                Self::Aggregate(aggregate) => aggregate.into_stream(txn).await,
                Self::Collection(collection) => collection.into_stream(txn).await,
                Self::Filter(filter) => filter.into_stream(txn).await,
                Self::Flatten(source) => {
                    source
                        .into_stream(txn.clone())
                        .map_ok(|source| Self::execute_flatten(source, txn))
                        .await
                }
                Self::Map(map) => map.into_stream(txn).await,
                Self::Range(start, stop, step) => {
                    let range = RangeStream::new(start, stop, step)
                        .map(Value::Number)
                        .map(State::from)
                        .map(Ok);
                    let range: TCBoxTryStream<_> = Box::pin(range);
                    Ok(range)
                }
            }
        })
    }

    fn execute_flatten(
        source: TCBoxTryStream<'static, State>,
        txn: Txn,
    ) -> TCBoxTryStream<'static, State> {
        let flat = source
            .map(|result| {
                result.and_then(|state| {
                    TCStream::try_cast_from(state, |s| {
                        TCError::bad_request("Stream::flatten expects a Stream, not ", s)
                    })
                })
            })
            .map_ok(move |stream| stream.into_stream(txn.clone()))
            .try_buffered(num_cpus::get())
            .try_flatten();

        Box::pin(flat)
    }
}

#[async_trait]
impl<'en> IntoView<'en, fs::Dir> for TCStream {
    type Txn = Txn;
    type View = en::SeqStream<TCResult<StateView<'en>>, TCBoxTryStream<'en, StateView<'en>>>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        let stream = self.into_stream(txn.clone()).await?;
        let view_stream: TCBoxTryStream<'en, StateView<'en>> = Box::pin(
            stream
                .map_ok(move |state| state.into_view(txn.clone()))
                .try_buffered(num_cpus::get()),
        );

        Ok(en::SeqStream::from(view_stream))
    }
}

impl<T> From<T> for TCStream
where
    crate::collection::Collection: From<T>,
{
    fn from(collection: T) -> Self {
        Self::Collection(crate::collection::Collection::from(collection).into())
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
            step,
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
