use std::collections::HashMap;

use async_trait::async_trait;
use destream::{en, EncodeMap};
use futures::stream::{self, StreamExt, TryStreamExt};
use futures::TryFutureExt;

use tc_error::*;
use tc_transact::IntoView;
use tcgeneric::{Id, NativeClass, TCBoxTryStream};

use crate::chain::ChainView;
use crate::collection::CollectionView;
use crate::fs;
use crate::object::ObjectView;
use crate::scalar::{OpDef, Scalar};
use crate::state::StateType;
use crate::txn::Txn;

use super::State;

/// A view of a [`State`] within a single [`Txn`], used for serialization.
pub enum StateView<'en> {
    Chain(ChainView<'en, CollectionView<'en>>),
    Closure((HashMap<Id, StateView<'en>>, OpDef)),
    Collection(CollectionView<'en>),
    Map(HashMap<Id, StateView<'en>>),
    Object(Box<ObjectView>),
    Scalar(Scalar),
    Stream(en::SeqStream<TCResult<StateView<'en>>, TCBoxTryStream<'en, StateView<'en>>>),
    Tuple(Vec<StateView<'en>>),
}

#[async_trait]
impl<'en> IntoView<'en, fs::Dir> for State {
    type Txn = Txn;
    type View = StateView<'en>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        match self {
            Self::Chain(chain) => chain.into_view(txn).map_ok(StateView::Chain).await,
            Self::Closure(closure) => closure.into_view(txn).map_ok(StateView::Closure).await,
            Self::Collection(collection) => {
                collection
                    .into_view(txn)
                    .map_ok(StateView::Collection)
                    .await
            }
            Self::Map(map) => {
                let map_view = stream::iter(map.into_iter())
                    .map(|(key, state)| state.into_view(txn.clone()).map_ok(|view| (key, view)))
                    .buffer_unordered(num_cpus::get())
                    .try_collect::<HashMap<Id, StateView>>()
                    .await?;

                Ok(StateView::Map(map_view))
            }
            Self::Object(object) => {
                object
                    .into_view(txn)
                    .map_ok(Box::new)
                    .map_ok(StateView::Object)
                    .await
            }
            Self::Scalar(scalar) => Ok(StateView::Scalar(scalar)),
            Self::Stream(stream) => stream.into_view(txn).map_ok(StateView::Stream).await,
            Self::Tuple(tuple) => {
                let tuple_view = stream::iter(tuple.into_iter())
                    .map(|state| state.into_view(txn.clone()))
                    .buffered(num_cpus::get())
                    .try_collect::<Vec<StateView>>()
                    .await?;

                Ok(StateView::Tuple(tuple_view.into()))
            }
        }
    }
}

impl<'en> en::IntoStream<'en> for StateView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Chain(chain) => chain.into_stream(encoder),
            Self::Closure(closure) => {
                let mut map = encoder.encode_map(Some(1))?;
                map.encode_key(StateType::Closure.path().to_string())?;
                map.encode_value(closure)?;
                map.end()
            }
            Self::Collection(collection) => collection.into_stream(encoder),
            Self::Map(map) => map.into_stream(encoder),
            Self::Object(object) => object.into_stream(encoder),
            Self::Scalar(scalar) => scalar.into_stream(encoder),
            Self::Stream(stream) => stream.into_stream(encoder),
            Self::Tuple(tuple) => tuple.into_stream(encoder),
        }
    }
}
