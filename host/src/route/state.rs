use safecast::TryCastFrom;
use std::convert::TryInto;

use tc_error::*;
use tcgeneric::{path_label, Id, Instance, NativeClass, PathLabel, PathSegment, Tuple};

use crate::scalar::Link;
use crate::state::{State, StateType};

use super::*;

const CLASS: PathLabel = path_label(&["class"]);

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

struct SelfHandler<'a> {
    subject: &'a State,
}

impl<'a> Handler<'a> for SelfHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    Ok(self.subject.clone())
                } else {
                    Err(TCError::not_found(key))
                }
            })
        }))
    }
}

impl Route for State {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        let child_handler = match self {
            Self::Collection(collection) => collection.route(path),
            Self::Chain(chain) => chain.route(path),
            Self::Map(map) => map.route(path),
            Self::Object(object) => object.route(path),
            Self::Scalar(scalar) => scalar.route(path),
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
