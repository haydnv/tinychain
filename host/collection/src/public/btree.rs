//! Public API endpoints for a [`Collection`]

use freqfs::FileSave;
use safecast::{AsType, CastFrom, TryCastFrom};

use tc_error::TCError;
use tc_scalar::Scalar;
use tc_transact::fs;
use tc_transact::public::{GetHandler, Handler, Route, StateInstance};
use tc_value::Value;
use tcgeneric::{Map, PathSegment, TCPath, Tuple};

use crate::btree::{BTree, BTreeFile, BTreeInstance, BTreeSchema, Node as BTreeNode};
use crate::{Collection, CollectionBase, CollectionType, Schema};

impl<State> Route<State> for CollectionType
where
    State: StateInstance + From<Collection<State::Txn, State::FE>>,
    State::FE: for<'a> FileSave<'a> + AsType<BTreeNode>,
    Collection<State::Txn, State::FE>: TryCastFrom<State>,
    Value: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        match self {
            Self::Null => None,
            Self::BTree(btt) => btt.route(path),
        }
    }
}

struct SchemaHandler<'a, Txn, FE> {
    collection: &'a Collection<Txn, FE>,
}

impl<'a, State> Handler<'a, State> for SchemaHandler<'a, State::Txn, State::FE>
where
    State: StateInstance,
    State::FE: AsType<BTreeNode>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                key.expect_none()?;

                let schema = match self.collection {
                    Collection::Null(_, _) => Schema::Null,
                    Collection::BTree(btree) => btree.schema().clone().into(),
                };

                Ok(Value::cast_from(schema).into())
            })
        }))
    }
}

impl<'a, Txn, FE> From<&'a Collection<Txn, FE>> for SchemaHandler<'a, Txn, FE> {
    fn from(collection: &'a Collection<Txn, FE>) -> Self {
        Self { collection }
    }
}

impl<State> Route<State> for Collection<State::Txn, State::FE>
where
    State: StateInstance + From<Collection<State::Txn, State::FE>> + From<Tuple<Value>> + From<u64>,
    State::FE: for<'a> FileSave<'a> + AsType<BTreeNode>,
    BTree<State::Txn, State::FE>: TryCastFrom<State>,
    Map<Value>: TryFrom<State, Error = TCError>,
    Tuple<State>: TryFrom<State, Error = TCError>,
    Scalar: TryCastFrom<State>,
    Value: TryCastFrom<State>,
    bool: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        log::debug!("Collection::route {}", TCPath::from(path));

        let child_handler: Option<Box<dyn Handler<'a, State> + 'a>> = match self {
            Self::Null(_, _) => None,
            Self::BTree(btree) => btree.route(path),
        };

        if child_handler.is_some() {
            return child_handler;
        }

        if path.len() == 1 {
            match path[0].as_str() {
                "schema" => Some(Box::new(SchemaHandler::from(self))),
                _ => None,
            }
        } else {
            None
        }
    }
}

impl<State> Route<State> for CollectionBase<State::Txn, State::FE>
where
    State: StateInstance + From<Collection<State::Txn, State::FE>> + From<Tuple<Value>> + From<u64>,
    State::FE: for<'a> FileSave<'a> + AsType<BTreeNode>,
    BTree<State::Txn, State::FE>: TryCastFrom<State>,
    Collection<State::Txn, State::FE>: From<BTree<State::Txn, State::FE>>,
    Map<Value>: TryFrom<State, Error = TCError>,
    Scalar: TryCastFrom<State>,
    Tuple<State>: TryFrom<State, Error = TCError>,
    Value: TryCastFrom<State>,
    bool: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        match self {
            Self::Null(_, _) => None,
            Self::BTree(btree) => btree.route(path),
        }
    }
}

pub struct Static;

impl<State> Route<State> for Static
where
    State: StateInstance + From<Collection<State::Txn, State::FE>> + From<Tuple<Value>>,
    State::FE: for<'a> FileSave<'a> + AsType<BTreeNode>,
    BTreeFile<State::Txn, State::FE>:
        fs::Persist<State::FE, Schema = BTreeSchema, Txn = State::Txn>,
    Collection<State::Txn, State::FE>: From<BTreeFile<State::Txn, State::FE>> + TryCastFrom<State>,
    Value: TryCastFrom<State>,
    bool: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        if path.is_empty() {
            return None;
        }

        match path[0].as_str() {
            "btree" => crate::btree::public::Static.route(&path[1..]),
            _ => None,
        }
    }
}
