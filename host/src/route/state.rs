use tc_error::*;
use tcgeneric::{path_label, Instance, NativeClass, PathLabel, PathSegment};

use crate::scalar::Link;
use crate::state::State;

use super::*;

const CLASS: PathLabel = path_label(&["class"]);

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

struct ClassHandler<'a> {
    subject: &'a State,
}

impl<'a> Handler<'a> for ClassHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, _key| {
            Box::pin(async move { Ok(Link::from(self.subject.class().path()).into()) })
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
            Some(Box::new(ClassHandler { subject: self }))
        } else {
            None
        }
    }
}
