use safecast::AsType;
use tc_chain::ChainBlock;
use tc_collection::{BTreeNode, DenseCacheFile, TensorNode};
use tc_transact::public::generic::COPY;
use tc_transact::public::helpers::AttributeHandler;
use tc_transact::public::{GetHandler, Handler, PostHandler, Route};
use tc_transact::{fs, RPCClient, Transaction};
use tcgeneric::PathSegment;

use crate::object::{InstanceClass, InstanceExt, Object, ObjectType};
use crate::State;

mod instance;
pub mod method;

impl<Txn, FE> Route<State<Txn, FE>> for ObjectType {
    fn route<'a>(
        &'a self,
        _path: &'a [PathSegment],
    ) -> Option<Box<dyn Handler<'a, State<Txn, FE>> + 'a>> {
        None
    }
}

struct ClassHandler<'a> {
    class: &'a InstanceClass,
}

impl<'a, Txn, FE> Handler<'a, State<Txn, FE>> for ClassHandler<'a>
where
    Txn: Transaction<FE> + RPCClient<State<Txn, FE>>,
    FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'b> fs::FileSave<'b>
        + Clone,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State<Txn, FE>>>
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

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State<Txn, FE>>>
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

impl<Txn, FE> Route<State<Txn, FE>> for InstanceClass
where
    Txn: Transaction<FE> + RPCClient<State<Txn, FE>>,
    FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'a> fs::FileSave<'a>
        + Clone,
{
    fn route<'a>(
        &'a self,
        path: &'a [PathSegment],
    ) -> Option<Box<dyn Handler<'a, State<Txn, FE>> + 'a>> {
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

impl<Txn, FE> Route<State<Txn, FE>> for Object<Txn, FE>
where
    Txn: Transaction<FE> + RPCClient<State<Txn, FE>>,
    FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'a> fs::FileSave<'a>
        + Clone,
{
    fn route<'a>(
        &'a self,
        path: &'a [PathSegment],
    ) -> Option<Box<dyn Handler<'a, State<Txn, FE>> + 'a>> {
        match self {
            Self::Class(class) => class.route(path),
            Self::Instance(instance) => instance.route(path),
        }
    }
}
