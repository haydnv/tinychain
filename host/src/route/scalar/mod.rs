use log::debug;

use tc_error::bad_request;
use tc_value::{Bound, Range, Value};
use tcgeneric::PathSegment;

use crate::scalar::{Scalar, ScalarType};
use crate::state::State;

use super::{AttributeHandler, EchoHandler, GetHandler, Handler, Route, COPY};

mod cluster;
mod value;

struct CastHandler {
    class: ScalarType,
}

impl<'a> Handler<'a> for CastHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                debug!("cast {} into {}", key, self.class);

                Scalar::Value(key)
                    .into_type(self.class)
                    .map(State::Scalar)
                    .ok_or_else(|| bad_request!("cannot cast into {}", self.class))
            })
        }))
    }
}

impl Route for ScalarType {
    fn route<'a>(&'a self, _path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        None
    }
}

impl Route for Range {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            None
        } else {
            match path[0].as_str() {
                "start" => match &self.start {
                    Bound::In(value) => value.route(&path[1..]),
                    Bound::Ex(value) => value.route(&path[1..]),
                    Bound::Un => Value::None.route(&path[1..]),
                },
                "end" => match &self.end {
                    Bound::In(value) => value.route(&path[1..]),
                    Bound::Ex(value) => value.route(&path[1..]),
                    Bound::Un => Value::None.route(&path[1..]),
                },
                _ => None,
            }
        }
    }
}

impl Route for Scalar {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path == &COPY[..] {
            return Some(Box::new(AttributeHandler::from(self.clone())));
        }

        match self {
            Self::Cluster(cluster) => cluster.route(path),
            Self::Map(map) => map.route(path),
            Self::Op(op_def) if path.is_empty() => Some(Box::new(op_def.clone())),
            Self::Range(range) => range.route(path),
            Self::Ref(_) => None,
            Self::Value(value) => value.route(path),
            Self::Tuple(tuple) => tuple.route(path),
            _ => None,
        }
    }
}

pub(super) struct Static;

impl Route for Static {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            return Some(Box::new(EchoHandler));
        }

        if path[0] == value::PREFIX {
            value::Static.route(&path[1..])
        } else {
            None
        }
    }
}
