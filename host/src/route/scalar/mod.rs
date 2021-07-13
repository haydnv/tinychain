use tc_error::TCError;
use tc_value::{Bound, Range};
use tcgeneric::{label, Label, PathSegment};

use crate::scalar::{Scalar, ScalarType, Value};
use crate::state::{State, StateType};

use super::{GetHandler, Handler, Route};

mod op;
mod value;

pub const PREFIX: Label = label("scalar");

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
                let err = format!("cannot cast into {} from {}", self.class, key);
                State::Scalar(Scalar::Value(key))
                    .into_type(StateType::Scalar(self.class))
                    .ok_or_else(|| TCError::unsupported(err))
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
        match self {
            Self::Map(map) => map.route(path),
            Self::Op(op_def) => op_def.route(path),
            Self::Range(range) => range.route(path),
            Self::Ref(_) => None,
            Self::Value(value) => value.route(path),
            Self::Tuple(tuple) => tuple.route(path),
        }
    }
}

pub struct Static;

impl Route for Static {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            return None;
        }

        if path[0] == value::PREFIX {
            value::Static.route(&path[1..])
        } else {
            None
        }
    }
}
