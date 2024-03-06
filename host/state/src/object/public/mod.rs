use tc_transact::public::generic::COPY;
use tc_transact::public::helpers::AttributeHandler;
use tc_transact::public::{GetHandler, Handler, PostHandler, Route};
use tc_transact::{Gateway, Transaction};
use tcgeneric::PathSegment;

use crate::object::{InstanceClass, InstanceExt, Object, ObjectType};
use crate::{CacheBlock, State};

mod instance;
pub mod method;

impl<Txn> Route<State<Txn>> for ObjectType {
    fn route<'a>(
        &'a self,
        _path: &'a [PathSegment],
    ) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
        None
    }
}

struct ClassHandler<'a> {
    class: &'a InstanceClass,
}

impl<'a, Txn> Handler<'a, State<Txn>> for ClassHandler<'a>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State<Txn>>>
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

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State<Txn>>>
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

impl<Txn> Route<State<Txn>> for InstanceClass
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
{
    fn route<'a>(
        &'a self,
        path: &'a [PathSegment],
    ) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
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

impl<Txn> Route<State<Txn>> for Object<Txn>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
{
    fn route<'a>(
        &'a self,
        path: &'a [PathSegment],
    ) -> Option<Box<dyn Handler<'a, State<Txn>> + 'a>> {
        match self {
            Self::Class(class) => class.route(path),
            Self::Instance(instance) => instance.route(path),
        }
    }
}
