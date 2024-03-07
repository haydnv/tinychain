//! Public API endpoints for a [`Collection`]

#[cfg(all(feature = "btree", not(feature = "table")))]
mod btree;
#[cfg(all(feature = "table", not(feature = "tensor")))]
mod table;
#[cfg(feature = "tensor")]
mod tensor;

#[cfg(all(feature = "btree", not(feature = "table")))]
pub use btree::Static;
#[cfg(all(feature = "table", not(feature = "tensor")))]
pub use table::Static;
#[cfg(feature = "tensor")]
pub use tensor::Static;

#[cfg(not(feature = "btree"))]
impl<State> tc_transact::public::Route<State> for super::Collection<State::Txn, State::FE>
where
    State: tc_transact::public::StateInstance,
{
    fn route<'a>(
        &'a self,
        _path: &'a [tcgeneric::PathSegment],
    ) -> Option<Box<dyn tc_transact::public::Handler<State> + 'a>> {
        None
    }
}

#[cfg(not(feature = "btree"))]
impl<State> tc_transact::public::Route<State> for super::CollectionBase<State::Txn, State::FE>
where
    State: tc_transact::public::StateInstance,
{
    fn route<'a>(
        &'a self,
        _path: &'a [tcgeneric::PathSegment],
    ) -> Option<Box<dyn tc_transact::public::Handler<State> + 'a>> {
        None
    }
}

#[cfg(not(feature = "btree"))]
impl<State> tc_transact::public::Route<State> for super::CollectionType
where
    State: tc_transact::public::StateInstance,
{
    fn route<'a>(
        &'a self,
        _path: &'a [tcgeneric::PathSegment],
    ) -> Option<Box<dyn tc_transact::public::Handler<State> + 'a>> {
        None
    }
}

#[cfg(not(feature = "btree"))]
pub struct Static;

#[cfg(not(feature = "btree"))]
impl<State> tc_transact::public::Route<State> for Static {
    fn route<'a>(
        &'a self,
        _path: &'a [tcgeneric::PathSegment],
    ) -> Option<Box<dyn tc_transact::public::Handler<State> + 'a>> {
        None
    }
}
