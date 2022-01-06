//! A stream generator such as a `Collection` or a mapping or aggregation of its items

use std::convert::TryInto;

use async_trait::async_trait;
use destream::en;
use futures::future;
use futures::stream::TryStreamExt;
use log::debug;
use sha2::digest::{Digest, Output};
use sha2::Sha256;

use tc_error::*;
use tc_transact::IntoView;
use tc_value::Number;
use tcgeneric::{Id, TCBoxTryStream};

use crate::closure::Closure;
use crate::fs;
use crate::state::{State, StateView};
use crate::txn::Txn;

use group::Aggregate;
use range::Range;
use source::*;

pub use source::Source;

mod group;
mod range;
mod source;

/// A stream generator such as a `Collection` or a mapping or aggregation of its items
#[derive(Clone)]
pub enum TCStream {
    Aggregate(Box<Aggregate>),
    Collection(Collection),
    Filter(Box<Filter>),
    Flatten(Box<Flatten>),
    Map(Box<Map>),
    Range(Range),
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
        Flatten::new(self).into()
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

    /// Compute the SHA256 hash of this `TCStream`.
    pub async fn hash(self, txn: Txn) -> TCResult<Output<Sha256>> {
        let stream = self.into_stream(txn.clone()).await?;
        let mut hashes = stream
            .map_ok(|state| state.hash(txn.clone()))
            .try_buffered(num_cpus::get());

        let mut hasher = Sha256::default();
        while let Some(hash) = hashes.try_next().await? {
            hasher.update(&hash);
        }
        Ok(hasher.finalize())
    }

    /// Return a `TCStream` produced by calling the given [`Closure`] on each item in this stream.
    pub fn map(self, op: Closure) -> Self {
        Map::new(self, op).into()
    }

    /// Return a `TCStream` of numbers at the given `step` within the given range.
    pub fn range(start: Number, stop: Number, step: Number) -> Self {
        Range::new(start, stop, step).into()
    }
}

#[async_trait]
impl Source for TCStream {
    async fn into_stream(self, txn: Txn) -> TCResult<TCBoxTryStream<'static, State>> {
        match self {
            Self::Aggregate(aggregate) => aggregate.into_stream(txn).await,
            Self::Collection(collection) => collection.into_stream(txn).await,
            Self::Filter(filter) => filter.into_stream(txn).await,
            Self::Flatten(source) => source.into_stream(txn).await,
            Self::Map(map) => map.into_stream(txn).await,
            Self::Range(range) => range.into_stream(txn).await,
        }
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
