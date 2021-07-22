use std::convert::TryInto;

use safecast::{TryCastFrom, TryCastInto};
use tc_error::*;
use tcgeneric::{
    label, path_label, Id, Instance, Label, NativeClass, PathLabel, PathSegment, Tuple,
};

use crate::object::{InstanceClass, Object};
use crate::scalar::Link;
use crate::state::{State, StateType};

use super::*;

const CLASS: PathLabel = path_label(&["class"]);
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
                        TCError::bad_request("invalid member for object prototype", s)
                    })?;

                    proto.insert(id, member);
                }

                let class = InstanceClass::new(Some(self.class.path().into()), proto);
                Ok(Object::Class(class).into())
            })
        }))
    }
}

impl From<StateType> for ClassHandler {
    fn from(class: StateType) -> Self {
        Self { class }
    }
}

struct MapHandler;

impl<'a> Handler<'a> for MapHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let value = Tuple::<(Id, Value)>::try_cast_from(key, |v| {
                    TCError::bad_request("invalid Map", v)
                })?;

                let map = value
                    .into_iter()
                    .map(|(id, value)| (id, State::from(value)))
                    .collect();

                Ok(State::Map(map))
            })
        }))
    }
}

struct TupleHandler;

impl<'a> Handler<'a> for TupleHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let value: Tuple<Value> = key.try_into()?;
                Ok(State::Tuple(value.into_iter().map(State::from).collect()))
            })
        }))
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
        let child_handler = match self {
            Self::Chain(chain) => chain.route(path),
            Self::Closure(closure) => closure.route(path),
            Self::Collection(collection) => collection.route(path),
            Self::Map(map) => map.route(path),
            Self::Object(object) => object.route(path),
            Self::Scalar(scalar) => scalar.route(path),
            Self::Stream(stream) => stream.route(path),
            Self::Tuple(tuple) => tuple.route(path),
        };

        if let Some(handler) = child_handler {
            return Some(handler);
        }

        if path.is_empty() {
            Some(Box::new(SelfHandler { subject: self }))
        } else if path == &CLASS[..] {
            Some(Box::new(ClassHandler::from(self.class())))
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

        if path[0] == collection::PREFIX {
            collection::Static.route(&path[1..])
        } else if path[0] == scalar::PREFIX {
            scalar::Static.route(&path[1..])
        } else {
            None
        }
    }
}
