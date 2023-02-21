use futures::TryFutureExt;
use log::debug;
use safecast::TryCastInto;

use tc_error::*;
use tc_transact::AsyncHash;
use tc_value::{Link, Number};
use tcgeneric::{label, Id, Instance, Label, NativeClass, PathSegment};

use crate::object::{InstanceClass, Object};
use crate::state::{State, StateType};

use super::*;

pub const PREFIX: Label = label("state");

struct ClassHandler {
    class: StateType,
}

impl<'a> Handler<'a> for ClassHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, _key| {
            Box::pin(async move { Ok(Link::from(self.class.path()).into()) })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, params| {
            Box::pin(async move {
                let mut proto = Map::new();
                for (id, member) in params.into_iter() {
                    let member = member.try_cast_into(|s| {
                        TCError::invalid_value(s, "an attribute in an object prototype")
                    })?;

                    proto.insert(id, member);
                }

                let class = InstanceClass::extend(self.class.path().clone(), proto);
                Ok(Object::Class(class).into())
            })
        }))
    }
}

struct HashHandler {
    state: State,
}

impl<'a> Handler<'a> for HashHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                key.expect_none()?;

                self.state
                    .hash(txn)
                    .map_ok(Id::from_hash)
                    .map_ok(Value::from)
                    .map_ok(State::from)
                    .await
            })
        }))
    }
}

impl From<State> for HashHandler {
    fn from(state: State) -> HashHandler {
        Self { state }
    }
}

impl From<StateType> for ClassHandler {
    fn from(class: StateType) -> Self {
        Self { class }
    }
}

impl Route for StateType {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        let child_handler = match self {
            Self::Chain(ct) => ct.route(path),
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

impl Route for State {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        debug!("instance of {} route {}", self.class(), TCPath::from(path));

        if let Some(handler) = match self {
            Self::Chain(chain) => chain.route(path),
            Self::Closure(closure) if path.is_empty() => {
                let handler: Box<dyn Handler<'a> + 'a> = Box::new(closure.clone());
                Some(handler)
            }
            Self::Collection(collection) => collection.route(path),
            Self::Map(map) => map.route(path),
            Self::Object(object) => object.route(path),
            Self::Scalar(scalar) => scalar.route(path),
            Self::Stream(stream) => stream.route(path),
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

pub struct Static;

impl Route for Static {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            return Some(Box::new(EchoHandler));
        }

        match path[0].as_str() {
            #[cfg(any(feature = "btree", feature = "table", feature = "tensor"))]
            "collection" => collection::Static.route(&path[1..]),

            "scalar" => scalar::Static.route(&path[1..]),
            "map" => generic::MapStatic.route(&path[1..]),
            "stream" => stream::Static.route(&path[1..]),
            "tuple" => generic::TupleStatic.route(&path[1..]),

            _ => None,
        }
    }
}
