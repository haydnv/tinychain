use freqfs::FileSave;
use safecast::{AsType, CastFrom, TryCastFrom};

use tc_error::TCError;
use tc_scalar::Scalar;
use tc_transact::fs;
use tc_transact::public::{GetHandler, Handler, Route, StateInstance};
use tc_value::{Number, NumberType, Value};
use tcgeneric::{Map, PathSegment, TCPath, Tuple};

use crate::btree::{BTree, BTreeFile, BTreeInstance, BTreeSchema, Node as BTreeNode};
use crate::table::{TableFile, TableInstance, TableSchema};
use crate::tensor::{DenseCacheFile, Node as TensorNode, Tensor, TensorBase, TensorInstance};
use crate::{Collection, CollectionBase, CollectionType, Schema};

impl<State> Route<State> for CollectionType
where
    State: StateInstance
        + From<Collection<State::Txn, State::FE>>
        + From<Tensor<State::Txn, State::FE>>,
    State::FE: for<'a> FileSave<'a> + DenseCacheFile + AsType<BTreeNode> + AsType<TensorNode>,
    Collection<State::Txn, State::FE>: TryCastFrom<State>,
    TableFile<State::Txn, State::FE>:
        fs::Persist<State::FE, Schema = TableSchema, Txn = State::Txn>,
    Number: TryCastFrom<State>,
    Tensor<State::Txn, State::FE>: TryCastFrom<State>,
    Vec<Tensor<State::Txn, State::FE>>: TryCastFrom<State>,
    Value: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        match self {
            Self::Null => None,
            Self::BTree(btt) => btt.route(path),
            Self::Table(tt) => tt.route(path),
            Self::Tensor(tt) => tt.route(path),
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
                    Collection::Table(table) => table.schema().clone().into(),
                    Collection::Tensor(tensor) => match tensor {
                        Tensor::Dense(dense) => Schema::Dense(dense.schema().clone()),
                        Tensor::Sparse(sparse) => Schema::Sparse(sparse.schema().clone()),
                    },
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
    State: StateInstance
        + From<Collection<State::Txn, State::FE>>
        + From<Tensor<State::Txn, State::FE>>
        + From<Tuple<Value>>
        + From<u64>,
    State::Class: From<NumberType>,
    State::FE: for<'a> FileSave<'a> + DenseCacheFile + AsType<BTreeNode> + AsType<TensorNode>,
    BTree<State::Txn, State::FE>: TryCastFrom<State>,
    Map<Value>: TryFrom<State, Error = TCError>,
    Number: TryCastFrom<State>,
    Tensor<State::Txn, State::FE>: TryCastFrom<State>,
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
            Self::Table(table) => table.route(path),
            Self::Tensor(tensor) => tensor.route(path),
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
    State: StateInstance
        + From<Collection<State::Txn, State::FE>>
        + From<Tensor<State::Txn, State::FE>>
        + From<Tuple<Value>>
        + From<u64>,
    State::Class: From<NumberType>,
    State::FE: for<'a> FileSave<'a> + DenseCacheFile + AsType<BTreeNode> + AsType<TensorNode>,
    BTree<State::Txn, State::FE>: TryCastFrom<State>,
    Collection<State::Txn, State::FE>: From<BTree<State::Txn, State::FE>>,
    Map<Value>: TryFrom<State, Error = TCError>,
    Number: TryCastFrom<State>,
    Scalar: TryCastFrom<State>,
    Tensor<State::Txn, State::FE>: TryCastFrom<State>,
    Tuple<State>: TryFrom<State, Error = TCError>,
    Value: TryCastFrom<State>,
    bool: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        match self {
            Self::Null(_, _) => None,
            Self::BTree(btree) => btree.route(path),
            Self::Table(table) => table.route(path),
            Self::Tensor(tensor) => match tensor {
                TensorBase::Dense(dense) => dense.route(path),
                TensorBase::Sparse(sparse) => sparse.route(path),
            },
        }
    }
}

pub struct Static;

impl<State> Route<State> for Static
where
    State: StateInstance
        + From<Collection<State::Txn, State::FE>>
        + From<Tensor<State::Txn, State::FE>>
        + From<Tuple<Value>>,
    State::Class: From<NumberType>,
    State::FE: for<'a> FileSave<'a> + DenseCacheFile + AsType<BTreeNode> + AsType<TensorNode>,
    BTreeFile<State::Txn, State::FE>:
        fs::Persist<State::FE, Schema = BTreeSchema, Txn = State::Txn>,
    TableFile<State::Txn, State::FE>:
        fs::Persist<State::FE, Schema = TableSchema, Txn = State::Txn>,
    Collection<State::Txn, State::FE>: From<BTreeFile<State::Txn, State::FE>> + TryCastFrom<State>,
    Number: TryCastFrom<State>,
    Tensor<State::Txn, State::FE>: TryCastFrom<State>,
    Value: TryCastFrom<State>,
    Vec<Tensor<State::Txn, State::FE>>: TryCastFrom<State>,
    bool: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        if path.is_empty() {
            return None;
        }

        match path[0].as_str() {
            "btree" => crate::btree::public::Static.route(&path[1..]),
            "table" => crate::table::public::Static.route(&path[1..]),
            "tensor" => crate::tensor::public::Static.route(&path[1..]),
            _ => None,
        }
    }
}
