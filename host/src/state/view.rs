use std::collections::HashMap;
use std::iter::FromIterator;

use async_trait::async_trait;
use destream::en;
use futures::stream::{self, StreamExt, TryStreamExt};
use futures::TryFutureExt;

use tc_error::*;
use tc_transact::IntoView;
use tcgeneric::Id;

use crate::chain::ChainView;
use crate::collection::CollectionView;
use crate::fs;
use crate::object::ObjectView;
use crate::scalar::Scalar;
use crate::txn::Txn;

use super::State;

/// A view of a [`State`] within a single [`Txn`], used for serialization.
pub enum StateView<'en> {
    Collection(CollectionView<'en>),
    Chain(ChainView<'en>),
    Map(HashMap<Id, StateView<'en>>),
    Object(Box<ObjectView<'en>>),
    Scalar(Scalar),
    Tuple(Vec<StateView<'en>>),
}

#[async_trait]
impl<'en> IntoView<'en, fs::Dir> for State {
    type Txn = Txn;
    type View = StateView<'en>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        match self {
            Self::Collection(collection) => {
                collection
                    .into_view(txn)
                    .map_ok(StateView::Collection)
                    .await
            }
            Self::Chain(chain) => chain.into_view(txn).map_ok(StateView::Chain).await,
            Self::Map(map) => {
                let map_view = stream::iter(map.into_iter())
                    .then(|(key, state)| state.into_view(txn.clone()).map_ok(|view| (key, view)))
                    .try_collect::<Vec<(Id, StateView)>>()
                    .await?;

                Ok(StateView::Map(HashMap::from_iter(map_view)))
            }
            Self::Object(object) => {
                object
                    .into_view(txn)
                    .map_ok(Box::new)
                    .map_ok(StateView::Object)
                    .await
            }
            Self::Scalar(scalar) => Ok(StateView::Scalar(scalar)),
            Self::Tuple(tuple) => {
                let tuple_view = stream::iter(tuple.into_iter())
                    .then(|state| state.into_view(txn.clone()))
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
            Self::Collection(collection) => collection.into_stream(encoder),
            Self::Chain(chain) => chain.into_stream(encoder),
            Self::Map(map) => map.into_stream(encoder),
            Self::Object(object) => object.into_stream(encoder),
            Self::Scalar(scalar) => scalar.into_stream(encoder),
            Self::Tuple(tuple) => tuple.into_stream(encoder),
        }
    }
}
