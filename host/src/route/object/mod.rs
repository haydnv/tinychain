use tc_error::TCError;
use tcgeneric::PathSegment;

use crate::object::{InstanceClass, Object};
use crate::state::State;

use super::{GetHandler, Handler, Route};

mod instance;

struct SelfHandler<'a> {
    class: &'a InstanceClass,
}

impl<'a> Handler<'a> for SelfHandler<'a> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    Ok(State::Object(self.class.clone().into()))
                } else {
                    Err(TCError::not_found(key))
                }
            })
        }))
    }
}

impl Route for InstanceClass {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            Some(Box::new(SelfHandler { class: self }))
        } else {
            None
        }
    }
}

impl Route for Object {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        match self {
            Self::Class(class) => class.route(path),
            Self::Instance(instance) => instance.route(path),
        }
    }
}
