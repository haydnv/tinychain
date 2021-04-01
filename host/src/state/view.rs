use destream::en;

use tc_transact::IntoView;
use tcgeneric::{Map, Tuple};

use crate::fs;
use crate::object::ObjectView;
use crate::scalar::Scalar;
use crate::txn::Txn;

use super::State;

#[derive(Clone)]
pub enum StateView {
    Map(Map<StateView>),
    Object(ObjectView),
    Scalar(Scalar),
    Tuple(Tuple<StateView>),
}

impl<'en> IntoView<'en, fs::Dir> for State {
    type Txn = Txn;
    type View = StateView;

    fn into_view(self, txn: Self::Txn) -> Self::View {
        match self {
            Self::Chain(_chain) => unimplemented!(),
            Self::Map(map) => {
                let map_view = map
                    .into_iter()
                    .map(|(key, state)| (key, state.into_view(txn.clone())))
                    .collect();

                StateView::Map(map_view)
            }
            Self::Object(object) => StateView::Object(object.into_view(txn)),
            Self::Scalar(scalar) => StateView::Scalar(scalar),
            Self::Tuple(tuple) => {
                let tuple_view = tuple
                    .into_iter()
                    .map(|state| state.into_view(txn.clone()))
                    .collect();

                StateView::Tuple(tuple_view)
            }
        }
    }
}

impl<'en> en::IntoStream<'en> for StateView {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            StateView::Map(map) => map.into_stream(encoder),
            StateView::Object(object) => object.into_stream(encoder),
            StateView::Scalar(scalar) => scalar.into_stream(encoder),
            StateView::Tuple(tuple) => tuple.into_stream(encoder),
        }
    }
}
