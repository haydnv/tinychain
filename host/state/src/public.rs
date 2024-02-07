use futures::TryFutureExt;
use log::debug;
use safecast::{AsType, TryCastInto};
use std::marker::PhantomData;
use tc_chain::ChainBlock;

use tc_collection::btree::{BTreeFile, BTreeSchema};
use tc_collection::table::{TableFile, TableSchema};
use tc_collection::{BTreeNode, Collection, DenseCacheFile, TensorNode};
use tc_error::*;
use tc_transact::public::helpers::{AttributeHandler, EchoHandler, SelfHandler};
use tc_transact::public::{GetHandler, Handler, PostHandler, Route};
use tc_transact::{fs, AsyncHash, RPCClient, Transaction};
use tc_value::{Link, Number, Value};
use tcgeneric::{label, Id, Instance, Label, Map, NativeClass, PathSegment, TCPath};

use crate::object::{InstanceClass, Object};
use crate::{State, StateType};

pub const PREFIX: Label = label("state");

struct ClassHandler {
    class: StateType,
}

impl<'a, Txn, FE> Handler<'a, State<Txn, FE>> for ClassHandler
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
        Some(Box::new(|_txn, _key| {
            Box::pin(async move { Ok(Link::from(self.class.path()).into()) })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State<Txn, FE>>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, params| {
            Box::pin(async move {
                let mut proto = Map::new();
                for (id, member) in params.into_iter() {
                    let member = member.try_cast_into(|s| {
                        TCError::unexpected(s, "an attribute in an public prototype")
                    })?;

                    proto.insert(id, member);
                }

                let class = InstanceClass::extend(self.class.path().clone(), proto);
                Ok(Object::Class(class).into())
            })
        }))
    }
}

struct HashHandler<Txn, FE> {
    state: State<Txn, FE>,
}

impl<'a, Txn, FE> Handler<'a, State<Txn, FE>> for HashHandler<Txn, FE>
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
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                key.expect_none()?;

                self.state
                    .hash(*txn.id())
                    .map_ok(Id::from)
                    .map_ok(Value::from)
                    .map_ok(State::from)
                    .await
            })
        }))
    }
}

impl<Txn, FE> From<State<Txn, FE>> for HashHandler<Txn, FE> {
    fn from(state: State<Txn, FE>) -> HashHandler<Txn, FE> {
        Self { state }
    }
}

impl From<StateType> for ClassHandler {
    fn from(class: StateType) -> Self {
        Self { class }
    }
}

impl<Txn, FE> Route<State<Txn, FE>> for StateType
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

impl<Txn, FE> Route<State<Txn, FE>> for State<Txn, FE>
where
    Txn: Transaction<FE> + RPCClient<Self>,
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
        debug!(
            "instance of {:?} route {}",
            self.class(),
            TCPath::from(path)
        );

        if let Some(handler) = match self {
            Self::Chain(chain) => chain.route(path),
            Self::Closure(closure) if path.is_empty() => {
                let handler: Box<dyn Handler<'a, State<Txn, FE>> + 'a> = Box::new(closure.clone());
                Some(handler)
            }
            Self::Collection(collection) => collection.route(path),
            Self::Map(map) => map.route(path),
            Self::Object(object) => object.route(path),
            Self::Scalar(scalar) => scalar.route(path),
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

#[derive(Copy, Clone)]
pub struct Static<Txn, FE> {
    phantom: PhantomData<(Txn, FE)>,
}

impl<Txn, FE> Default for Static<Txn, FE> {
    fn default() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<Txn, FE> Route<State<Txn, FE>> for Static<Txn, FE>
where
    Txn: Transaction<FE> + RPCClient<State<Txn, FE>>,
    FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'a> fs::FileSave<'a>
        + Clone,
    BTreeFile<Txn, FE>: fs::Persist<FE, Schema = BTreeSchema, Txn = Txn>,
    TableFile<Txn, FE>: fs::Persist<FE, Schema = TableSchema, Txn = Txn>,
    Collection<Txn, FE>: From<BTreeFile<Txn, FE>>,
{
    fn route<'a>(
        &'a self,
        path: &'a [PathSegment],
    ) -> Option<Box<dyn Handler<'a, State<Txn, FE>> + 'a>> {
        if path.is_empty() {
            return Some(Box::new(EchoHandler));
        }

        match path[0].as_str() {
            "collection" => tc_collection::public::Static.route(&path[1..]),
            "scalar" => tc_scalar::public::Static.route(&path[1..]),
            "map" => tc_transact::public::generic::MapStatic.route(&path[1..]),
            "tuple" => tc_transact::public::generic::TupleStatic.route(&path[1..]),
            _ => None,
        }
    }
}
