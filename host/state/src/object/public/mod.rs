use tc_transact::public::generic::COPY;
use tc_transact::public::helpers::AttributeHandler;
use tc_transact::public::{GetHandler, Handler, PostHandler, Route};
use tcgeneric::PathSegment;

use crate::object::{InstanceClass, InstanceExt, Object, ObjectType};
use crate::{State, Txn};

mod instance;
pub mod method;

impl Route<State> for ObjectType {
    fn route<'a>(&'a self, _path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        None
    }
}

struct ClassHandler<'a> {
    class: &'a InstanceClass,
}

impl<'a> Handler<'a, State> for ClassHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let parent = State::from(key);
                let instance = InstanceExt::new(parent, self.class.clone());
                Ok(State::Object(instance.into()))
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, members| {
            Box::pin(async move {
                let instance =
                    InstanceExt::anonymous(State::default(), self.class.clone(), members);

                Ok(State::Object(instance.into()))
            })
        }))
    }
}

impl Route<State> for InstanceClass {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        if path == &COPY[..] {
            return Some(Box::new(AttributeHandler::from(Object::Class(
                self.clone(),
            ))));
        }

        if path.is_empty() {
            Some(Box::new(ClassHandler { class: self }))
        } else if let Some(attribute) = self.proto().get(&path[0]) {
            attribute.route(&path[1..])
        } else {
            None
        }
    }
}

impl Route<State> for Object {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        match self {
            Self::Class(class) => class.route(path),
            Self::Instance(instance) => instance.route(path),
        }
    }
}
