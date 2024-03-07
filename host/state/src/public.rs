use std::marker::PhantomData;

use futures::TryFutureExt;
use log::debug;
use safecast::TryCastInto;

use tc_error::*;
#[cfg(any(feature = "btree", feature = "table"))]
use tc_transact::fs;
use tc_transact::hash::AsyncHash;
use tc_transact::public::helpers::{AttributeHandler, EchoHandler, SelfHandler};
use tc_transact::public::{GetHandler, Handler, PostHandler, Route};
use tc_transact::{Gateway, Transaction};
use tc_value::{Link, Number, Value};
use tcgeneric::{label, Id, Instance, Label, Map, NativeClass, PathSegment, TCPath};

#[cfg(any(feature = "btree", feature = "table"))]
use crate::collection::{BTreeFile, BTreeSchema};
#[cfg(feature = "table")]
use crate::collection::{TableFile, TableSchema};
use crate::object::{InstanceClass, Object};
#[cfg(any(feature = "btree", feature = "table"))]
use crate::Collection;
use crate::{CacheBlock, State, StateType};

pub const PREFIX: Label = label("state");

struct ClassHandler {
    class: StateType,
}

impl<'a, Txn> Handler<'a, State<Txn>> for ClassHandler
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State<Txn>>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, _key| {
            Box::pin(async move { Ok(Link::from(self.class.path()).into()) })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State<Txn>>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, params| {
            Box::pin(async move {
                let mut proto = Map::new();
                for (id, member) in params.into_iter() {
                    let member = member.try_cast_into(|s| {
                        TCError::unexpected(s, "an attribute in an public prototype")
                    })?;

                    proto.insert(id, member);
                }

                let class = InstanceClass::extend(self.class.path().clone(), proto);
                Ok(Object::Class(class).into())
            })
        }))
    }
}

struct HashHandler<Txn> {
    state: State<Txn>,
}

impl<'a, Txn> Handler<'a, State<Txn>> for HashHandler<Txn>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State<Txn>>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                key.expect_none()?;

                self.state
                    .hash(*txn.id())
                    .map_ok(Id::from)
                    .map_ok(Value::from)
                    .map_ok(State::from)
                    .await
            })
        }))
    }
}

impl<Txn> From<State<Txn>> for HashHandler<Txn> {
    fn from(state: State<Txn>) -> HashHandler<Txn> {
        Self { state }
    }
}

impl From<StateType> for ClassHandler {
    fn from(class: StateType) -> Self {
        Self { class }
    }
}

impl<Txn> Route<State<Txn>> for StateType
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
{
    fn route<'a>(
        &'a self,
        path: &'a [PathSegment],
    ) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
        let child_handler = match self {
            #[cfg(feature = "chain")]
            Self::Chain(ct) => ct.route(path),
            #[cfg(feature = "collection")]
            Self::Collection(ct) => ct.route(path),
            Self::Object(ot) => ot.route(path),
            Self::Scalar(st) => st.route(path),
            _ => None,
        };

        if child_handler.is_some() {
            return child_handler;
        }

        if path.is_empty() {
            Some(Box::new(ClassHandler::from(*self)))
        } else {
            None
        }
    }
}

impl<Txn> Route<State<Txn>> for State<Txn>
where
    Txn: Transaction<CacheBlock> + Gateway<Self>,
{
    fn route<'a>(
        &'a self,
        path: &'a [PathSegment],
    ) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
        debug!(
            "instance of {:?} route {}",
            self.class(),
            TCPath::from(path)
        );

        if let Some(handler) = match self {
            #[cfg(feature = "chain")]
            Self::Chain(chain) => chain.route(path),
            Self::Closure(closure) if path.is_empty() => {
                let handler: Box<dyn Handler<'a, State<Txn>> + 'a> = Box::new(closure.clone());
                Some(handler)
            }
            #[cfg(feature = "collection")]
            Self::Collection(collection) => collection.route(path),
            Self::Map(map) => map.route(path),
            Self::Object(object) => object.route(path),
            Self::Scalar(scalar) => scalar.route(path),
            Self::Tuple(tuple) => tuple.route(path),
            _ => None,
        } {
            return Some(handler);
        }

        if path.is_empty() {
            Some(Box::new(SelfHandler::from(self)))
        } else if path.len() == 1 {
            match path[0].as_str() {
                "class" => Some(Box::new(ClassHandler::from(self.class()))),
                "hash" => Some(Box::new(HashHandler::from(self.clone()))),
                "is_none" => Some(Box::new(AttributeHandler::from(Number::Bool(
                    self.is_none().into(),
                )))),
                _ => None,
            }
        } else {
            None
        }
    }
}

#[derive(Copy, Clone)]
pub struct Static<Txn> {
    phantom: PhantomData<Txn>,
}

impl<Txn> Default for Static<Txn> {
    fn default() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

#[cfg(all(feature = "collection", not(feature = "btree"), not(feature = "table")))]
impl<Txn> Route<State<Txn>> for Static<Txn>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
{
    fn route<'a>(
        &'a self,
        path: &'a [PathSegment],
    ) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
        if path.is_empty() {
            return Some(Box::new(EchoHandler));
        }

        match path[0].as_str() {
            "collection" => tc_collection::public::Static.route(&path[1..]),
            "scalar" => tc_scalar::public::Static.route(&path[1..]),
            "map" => tc_transact::public::generic::MapStatic.route(&path[1..]),
            "tuple" => tc_transact::public::generic::TupleStatic.route(&path[1..]),
            _ => None,
        }
    }
}

#[cfg(all(feature = "btree", not(feature = "table")))]
impl<Txn> Route<State<Txn>> for Static<Txn>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
    BTreeFile<Txn>: fs::Persist<CacheBlock, Schema = BTreeSchema, Txn = Txn>,
    Collection<Txn>: From<BTreeFile<Txn>>,
{
    fn route<'a>(
        &'a self,
        path: &'a [PathSegment],
    ) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
        if path.is_empty() {
            return Some(Box::new(EchoHandler));
        }

        match path[0].as_str() {
            "collection" => tc_collection::public::Static.route(&path[1..]),
            "scalar" => tc_scalar::public::Static.route(&path[1..]),
            "map" => tc_transact::public::generic::MapStatic.route(&path[1..]),
            "tuple" => tc_transact::public::generic::TupleStatic.route(&path[1..]),
            _ => None,
        }
    }
}

#[cfg(feature = "table")]
impl<Txn> Route<State<Txn>> for Static<Txn>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
    BTreeFile<Txn>: fs::Persist<CacheBlock, Schema = BTreeSchema, Txn = Txn>,
    TableFile<Txn>: fs::Persist<CacheBlock, Schema = TableSchema, Txn = Txn>,
    Collection<Txn>: From<BTreeFile<Txn>>,
{
    fn route<'a>(
        &'a self,
        path: &'a [PathSegment],
    ) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
        if path.is_empty() {
            return Some(Box::new(EchoHandler));
        }

        match path[0].as_str() {
            "collection" => tc_collection::public::Static.route(&path[1..]),
            "scalar" => tc_scalar::public::Static.route(&path[1..]),
            "map" => tc_transact::public::generic::MapStatic.route(&path[1..]),
            "tuple" => tc_transact::public::generic::TupleStatic.route(&path[1..]),
            _ => None,
        }
    }
}

#[cfg(not(feature = "collection"))]
impl<Txn> Route<State<Txn>> for Static<Txn>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
{
    fn route<'a>(
        &'a self,
        path: &'a [PathSegment],
    ) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
        if path.is_empty() {
            return Some(Box::new(EchoHandler));
        }

        match path[0].as_str() {
            "scalar" => tc_scalar::public::Static.route(&path[1..]),
            "map" => tc_transact::public::generic::MapStatic.route(&path[1..]),
            "tuple" => tc_transact::public::generic::TupleStatic.route(&path[1..]),
            _ => None,
        }
    }
}
