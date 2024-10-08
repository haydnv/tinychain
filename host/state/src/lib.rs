//! A TinyChain [`State`]

use std::collections::HashSet;
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::str::FromStr;

use async_trait::async_trait;
use bytes::Bytes;
use destream::de;
use destream::ArrayAccess;
use futures::future::TryFutureExt;
use futures::stream::{StreamExt, TryStreamExt};
use log::debug;
use safecast::*;

#[cfg(feature = "chain")]
use tc_chain::ChainVisitor;
#[cfg(feature = "collection")]
use tc_collection::{CollectionType, CollectionVisitor};
use tc_error::*;
use tc_scalar::*;
use tc_transact::hash::{AsyncHash, Hash, Output, Sha256};
use tc_transact::public::{ClosureInstance, Public, StateInstance, ToState};
use tc_transact::{Gateway, Transaction, TxnId};
use tc_value::{Float, Host, Link, Number, NumberType, TCString, Value, ValueType};
use tcgeneric::{
    label, path_label, Class, Id, Instance, Label, Map, NativeClass, PathLabel, PathSegment,
    TCPath, TCPathBuf, Tuple,
};

use closure::*;
use object::{InstanceClass, Object, ObjectType, ObjectVisitor};

pub use block::CacheBlock;
#[cfg(feature = "chain")]
pub use chain::*;
#[cfg(feature = "collection")]
pub use collection::*;

mod block;
pub mod closure;
pub mod object;
pub mod public;
pub mod view;

/// The path prefix of all [`State`] types.
pub const PREFIX: PathLabel = path_label(&["state"]);

#[cfg(feature = "chain")]
pub mod chain {
    use crate::{CacheBlock, State};

    pub use tc_chain::{ChainType, Recover};

    pub type Chain<Txn, T> = tc_chain::Chain<State<Txn>, Txn, CacheBlock, T>;
    pub type BlockChain<Txn, T> = tc_chain::BlockChain<State<Txn>, Txn, CacheBlock, T>;
    pub type SyncChain<Txn, T> = tc_chain::SyncChain<State<Txn>, Txn, CacheBlock, T>;
}

#[cfg(feature = "collection")]
pub mod collection {
    use crate::CacheBlock;

    pub use tc_collection::Schema;

    #[cfg(feature = "btree")]
    pub use tc_collection::btree::{BTreeSchema, BTreeType};
    #[cfg(all(feature = "table", not(feature = "btree")))]
    pub(crate) use tc_collection::btree::{BTreeSchema, BTreeType};
    #[cfg(feature = "table")]
    pub use tc_collection::table::{TableSchema, TableType};
    #[cfg(feature = "tensor")]
    pub use tc_collection::tensor::TensorType;

    pub type Collection<Txn> = tc_collection::Collection<Txn, CacheBlock>;
    pub type CollectionBase<Txn> = tc_collection::CollectionBase<Txn, CacheBlock>;

    #[cfg(feature = "btree")]
    pub type BTree<Txn> = tc_collection::BTree<Txn, CacheBlock>;
    #[cfg(feature = "btree")]
    pub type BTreeFile<Txn> = tc_collection::BTreeFile<Txn, CacheBlock>;
    #[cfg(all(any(feature = "table", feature = "tensor"), not(feature = "btree")))]
    pub(crate) type BTree<Txn> = tc_collection::BTree<Txn, CacheBlock>;
    #[cfg(all(feature = "table", not(feature = "btree")))]
    pub(crate) type BTreeFile<Txn> = tc_collection::BTreeFile<Txn, CacheBlock>;
    #[cfg(feature = "table")]
    pub type Table<Txn> = tc_collection::Table<Txn, CacheBlock>;
    #[cfg(feature = "table")]
    pub type TableFile<Txn> = tc_collection::TableFile<Txn, CacheBlock>;
    #[cfg(feature = "tensor")]
    pub type Tensor<Txn> = tc_collection::Tensor<Txn, CacheBlock>;
}

/// The [`Class`] of a [`State`].
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum StateType {
    #[cfg(feature = "chain")]
    Chain(ChainType),
    #[cfg(feature = "collection")]
    Collection(CollectionType),
    Closure,
    Map,
    Object(ObjectType),
    Scalar(ScalarType),
    Tuple,
}

impl Class for StateType {}

impl NativeClass for StateType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        debug!("StateType::from_path {}", TCPath::from(path));

        if path.is_empty() {
            None
        } else if &path[0] == "state" {
            if path.len() == 2 {
                match path[1].as_str() {
                    "closure" => Some(Self::Closure),
                    "map" => Some(Self::Map),
                    "tuple" => Some(Self::Tuple),
                    _ => None,
                }
            } else if path.len() > 2 {
                match path[1].as_str() {
                    #[cfg(feature = "chain")]
                    "chain" => ChainType::from_path(path).map(Self::Chain),
                    #[cfg(feature = "collection")]
                    "collection" => CollectionType::from_path(path).map(Self::Collection),
                    "object" => ObjectType::from_path(path).map(Self::Object),
                    "scalar" => ScalarType::from_path(path).map(Self::Scalar),
                    _ => None,
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        match self {
            #[cfg(feature = "chain")]
            Self::Chain(ct) => ct.path(),
            #[cfg(feature = "collection")]
            Self::Collection(ct) => ct.path(),
            Self::Closure => path_label(&["state", "closure"]).into(),
            Self::Map => path_label(&["state", "map"]).into(),
            Self::Object(ot) => ot.path(),
            Self::Scalar(st) => st.path(),
            Self::Tuple => path_label(&["state", "tuple"]).into(),
        }
    }
}

#[cfg(any(feature = "btree", feature = "table"))]
impl From<BTreeType> for StateType {
    fn from(btt: BTreeType) -> Self {
        CollectionType::BTree(btt).into()
    }
}

#[cfg(feature = "collection")]
impl From<CollectionType> for StateType {
    fn from(ct: CollectionType) -> Self {
        Self::Collection(ct)
    }
}

impl From<NumberType> for StateType {
    fn from(nt: NumberType) -> Self {
        ValueType::from(nt).into()
    }
}

#[cfg(feature = "chain")]
impl From<ChainType> for StateType {
    fn from(ct: ChainType) -> Self {
        Self::Chain(ct)
    }
}

impl From<ObjectType> for StateType {
    fn from(ot: ObjectType) -> Self {
        Self::Object(ot)
    }
}

impl From<ScalarType> for StateType {
    fn from(st: ScalarType) -> Self {
        Self::Scalar(st)
    }
}

#[cfg(feature = "table")]
impl From<TableType> for StateType {
    fn from(tt: TableType) -> Self {
        Self::Collection(tt.into())
    }
}

#[cfg(feature = "tensor")]
impl From<TensorType> for StateType {
    fn from(tt: TensorType) -> Self {
        Self::Collection(tt.into())
    }
}

impl From<ValueType> for StateType {
    fn from(vt: ValueType) -> Self {
        Self::Scalar(vt.into())
    }
}

impl TryFrom<StateType> for ScalarType {
    type Error = TCError;

    fn try_from(st: StateType) -> TCResult<Self> {
        match st {
            StateType::Scalar(st) => Ok(st),
            other => Err(TCError::unexpected(other, "a Scalar class")),
        }
    }
}

impl fmt::Debug for StateType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            #[cfg(feature = "chain")]
            Self::Chain(ct) => fmt::Debug::fmt(ct, f),
            #[cfg(feature = "collection")]
            Self::Collection(ct) => fmt::Debug::fmt(ct, f),
            Self::Closure => f.write_str("closure"),
            Self::Map => f.write_str("Map<State>"),
            Self::Object(ot) => fmt::Debug::fmt(ot, f),
            Self::Scalar(st) => fmt::Debug::fmt(st, f),
            Self::Tuple => f.write_str("Tuple<State>"),
        }
    }
}

/// An addressable state with a discrete value per-transaction.
pub enum State<Txn> {
    #[cfg(feature = "chain")]
    Chain(Chain<Txn, CollectionBase<Txn>>),
    #[cfg(feature = "collection")]
    Collection(Collection<Txn>),
    Closure(Closure<Txn>),
    Map(Map<Self>),
    Object(Object<Txn>),
    Scalar(Scalar),
    Tuple(Tuple<Self>),
}

impl<Txn> Clone for State<Txn> {
    fn clone(&self) -> Self {
        match self {
            #[cfg(feature = "chain")]
            Self::Chain(chain) => Self::Chain(chain.clone()),
            #[cfg(feature = "collection")]
            Self::Collection(collection) => Self::Collection(collection.clone()),
            Self::Closure(closure) => Self::Closure(closure.clone()),
            Self::Map(map) => Self::Map(map.clone()),
            Self::Object(obj) => Self::Object(obj.clone()),
            Self::Scalar(scalar) => Self::Scalar(scalar.clone()),
            Self::Tuple(tuple) => Self::Tuple(tuple.clone()),
        }
    }
}

impl<Txn> State<Txn> {
    // TODO: make this an associated const of the NativeClass trait
    pub const PREFIX: Label = label("state");

    /// Return true if this `State` is an empty [`Tuple`] or [`Map`], default [`Link`], or `Value::None`
    pub fn is_none(&self) -> bool {
        match self {
            Self::Map(map) => map.is_empty(),
            Self::Scalar(scalar) => scalar.is_none(),
            Self::Tuple(tuple) => tuple.is_empty(),
            _ => false,
        }
    }

    /// Return false if this `State` is an empty [`Tuple`] or [`Map`], default [`Link`], or `Value::None`
    pub fn is_some(&self) -> bool {
        !self.is_none()
    }

    /// Return this `State` as a [`Map`] of [`State`]s, or an error if this is not possible.
    pub fn try_into_map<Err, OnErr: Fn(Self) -> Err>(self, err: OnErr) -> Result<Map<Self>, Err> {
        match self {
            State::Map(states) => Ok(states),
            State::Scalar(Scalar::Map(states)) => Ok(states
                .into_iter()
                .map(|(id, scalar)| (id, State::Scalar(scalar)))
                .collect()),

            other => Err((err)(other)),
        }
    }

    /// Return this `State` as a [`Map`] of [`State`]s, or an error if this is not possible.
    // TODO: allow specifying an output type other than `State`
    pub fn try_into_tuple<Err: Fn(Self) -> TCError>(self, err: Err) -> TCResult<Tuple<Self>> {
        match self {
            State::Tuple(tuple) => Ok(tuple),
            State::Scalar(Scalar::Tuple(tuple)) => {
                Ok(tuple.into_iter().map(State::Scalar).collect())
            }
            State::Scalar(Scalar::Value(Value::Tuple(tuple))) => {
                Ok(tuple.into_iter().map(State::from).collect())
            }
            other => Err((err)(other)),
        }
    }
}

impl<Txn> State<Txn>
where
    Txn: Transaction<CacheBlock> + Gateway<Self>,
{
    /// Return true if this `State` is a reference that needs to be resolved.
    pub fn is_ref(&self) -> bool {
        match self {
            Self::Map(map) => map.values().any(Self::is_ref),
            Self::Scalar(scalar) => Refer::<Self>::is_ref(scalar),
            Self::Tuple(tuple) => tuple.iter().any(Self::is_ref),
            _ => false,
        }
    }
}

impl<Txn> StateInstance for State<Txn>
where
    Txn: Transaction<CacheBlock> + Gateway<Self>,
{
    type FE = CacheBlock;
    type Txn = Txn;
    type Closure = Closure<Txn>;

    fn is_map(&self) -> bool {
        match self {
            Self::Scalar(scalar) => scalar.is_map(),
            Self::Map(_) => true,
            _ => false,
        }
    }

    fn is_tuple(&self) -> bool {
        match self {
            Self::Scalar(scalar) => scalar.is_tuple(),
            Self::Tuple(_) => true,
            _ => false,
        }
    }
}

#[async_trait]
impl<Txn> Refer<Self> for State<Txn>
where
    Txn: Transaction<CacheBlock> + Gateway<Self>,
{
    fn dereference_self(self, path: &TCPathBuf) -> Self {
        match self {
            Self::Closure(closure) => Self::Closure(closure.dereference_self(path)),
            Self::Map(map) => {
                let map = map
                    .into_iter()
                    .map(|(id, state)| (id, state.dereference_self(path)))
                    .collect();

                Self::Map(map)
            }
            Self::Scalar(scalar) => Self::Scalar(Refer::<Self>::dereference_self(scalar, path)),
            Self::Tuple(tuple) => {
                let tuple = tuple
                    .into_iter()
                    .map(|state| state.dereference_self(path))
                    .collect();

                Self::Tuple(tuple)
            }
            other => other,
        }
    }

    fn is_conditional(&self) -> bool {
        match self {
            Self::Map(map) => map.values().any(|state| state.is_conditional()),
            Self::Scalar(scalar) => Refer::<Self>::is_conditional(scalar),
            Self::Tuple(tuple) => tuple.iter().any(|state| state.is_conditional()),
            _ => false,
        }
    }

    fn is_inter_service_write(&self, cluster_path: &[PathSegment]) -> bool {
        match self {
            Self::Closure(closure) => closure.is_inter_service_write(cluster_path),
            Self::Map(map) => map
                .values()
                .any(|state| state.is_inter_service_write(cluster_path)),

            Self::Scalar(scalar) => Refer::<Self>::is_inter_service_write(scalar, cluster_path),

            Self::Tuple(tuple) => tuple
                .iter()
                .any(|state| state.is_inter_service_write(cluster_path)),

            _ => false,
        }
    }

    fn is_ref(&self) -> bool {
        match self {
            Self::Map(map) => map.values().any(|state| state.is_ref()),
            Self::Scalar(scalar) => Refer::<Self>::is_ref(scalar),
            Self::Tuple(tuple) => tuple.iter().any(|state| state.is_ref()),
            _ => false,
        }
    }

    fn reference_self(self, path: &TCPathBuf) -> Self {
        match self {
            Self::Closure(closure) => Self::Closure(closure.reference_self(path)),
            Self::Map(map) => {
                let map = map
                    .into_iter()
                    .map(|(id, state)| (id, state.reference_self(path)))
                    .collect();

                Self::Map(map)
            }
            Self::Scalar(scalar) => Self::Scalar(Refer::<Self>::reference_self(scalar, path)),
            Self::Tuple(tuple) => {
                let tuple = tuple
                    .into_iter()
                    .map(|state| state.reference_self(path))
                    .collect();

                Self::Tuple(tuple)
            }
            other => other,
        }
    }

    fn requires(&self, deps: &mut HashSet<Id>) {
        match self {
            Self::Map(map) => {
                for state in map.values() {
                    state.requires(deps);
                }
            }
            Self::Scalar(scalar) => Refer::<Self>::requires(scalar, deps),
            Self::Tuple(tuple) => {
                for state in tuple.iter() {
                    state.requires(deps);
                }
            }
            _ => {}
        }
    }

    async fn resolve<'a, T: ToState<Self> + Instance + Public<Self>>(
        self,
        context: &'a Scope<'a, Self, T>,
        txn: &'a Txn,
    ) -> TCResult<Self> {
        debug!("State::resolve {:?}", self);

        match self {
            Self::Map(map) => {
                let mut resolved = futures::stream::iter(map)
                    .map(|(id, state)| state.resolve(context, txn).map_ok(|state| (id, state)))
                    .buffer_unordered(num_cpus::get());

                let mut map = Map::new();
                while let Some((id, state)) = resolved.try_next().await? {
                    map.insert(id, state);
                }

                Ok(State::Map(map))
            }
            Self::Scalar(scalar) => scalar.resolve(context, txn).await,
            Self::Tuple(tuple) => {
                let len = tuple.len();
                let mut resolved = futures::stream::iter(tuple)
                    .map(|state| state.resolve(context, txn))
                    .buffered(num_cpus::get());

                let mut tuple = Vec::with_capacity(len);
                while let Some(state) = resolved.try_next().await? {
                    tuple.push(state);
                }

                Ok(State::Tuple(tuple.into()))
            }
            other => Ok(other),
        }
    }
}

impl<Txn> Default for State<Txn> {
    fn default() -> Self {
        Self::Scalar(Scalar::default())
    }
}

impl<Txn> Instance for State<Txn>
where
    Txn: Send + Sync,
{
    type Class = StateType;

    fn class(&self) -> StateType {
        match self {
            #[cfg(feature = "chain")]
            Self::Chain(chain) => StateType::Chain(chain.class()),
            Self::Closure(_) => StateType::Closure,
            #[cfg(feature = "collection")]
            Self::Collection(collection) => StateType::Collection(collection.class()),
            Self::Map(_) => StateType::Map,
            Self::Object(object) => StateType::Object(object.class()),
            Self::Scalar(scalar) => StateType::Scalar(scalar.class()),
            Self::Tuple(_) => StateType::Tuple,
        }
    }
}

#[async_trait]
impl<Txn> AsyncHash for State<Txn>
where
    Txn: Transaction<CacheBlock> + Gateway<Self>,
{
    async fn hash(&self, txn_id: TxnId) -> TCResult<Output<Sha256>> {
        match self {
            #[cfg(feature = "chain")]
            Self::Chain(chain) => chain.hash(txn_id).await,
            Self::Closure(closure) => closure.hash(txn_id).await,
            #[cfg(feature = "collection")]
            Self::Collection(collection) => collection.hash(txn_id).await,
            Self::Map(map) => map.hash(txn_id).await,
            Self::Object(object) => object.hash(txn_id).await,
            Self::Scalar(scalar) => Ok(Hash::<Sha256>::hash(scalar)),
            Self::Tuple(tuple) => tuple.hash(txn_id).await,
        }
    }
}

impl<Txn> From<()> for State<Txn> {
    fn from(_: ()) -> Self {
        State::Scalar(Scalar::Value(Value::None))
    }
}

#[cfg(feature = "btree")]
impl<Txn> From<BTree<Txn>> for State<Txn> {
    fn from(btree: BTree<Txn>) -> Self {
        Self::Collection(btree.into())
    }
}

#[cfg(feature = "chain")]
impl<Txn> From<Chain<Txn, CollectionBase<Txn>>> for State<Txn> {
    fn from(chain: Chain<Txn, CollectionBase<Txn>>) -> Self {
        Self::Chain(chain)
    }
}

#[cfg(feature = "chain")]
impl<Txn> From<BlockChain<Txn, CollectionBase<Txn>>> for State<Txn> {
    fn from(chain: BlockChain<Txn, CollectionBase<Txn>>) -> Self {
        Self::Chain(chain.into())
    }
}

impl<Txn> From<Closure<Txn>> for State<Txn> {
    fn from(closure: Closure<Txn>) -> Self {
        Self::Closure(closure)
    }
}

#[cfg(feature = "collection")]
impl<Txn> From<Collection<Txn>> for State<Txn> {
    fn from(collection: Collection<Txn>) -> Self {
        Self::Collection(collection)
    }
}

#[cfg(feature = "collection")]
impl<Txn> From<CollectionBase<Txn>> for State<Txn> {
    fn from(collection: CollectionBase<Txn>) -> Self {
        Self::Collection(collection.into())
    }
}

impl<Txn> From<Id> for State<Txn> {
    fn from(id: Id) -> Self {
        Self::Scalar(id.into())
    }
}

impl<Txn> From<InstanceClass> for State<Txn> {
    fn from(class: InstanceClass) -> Self {
        Self::Object(class.into())
    }
}

impl<Txn> From<Host> for State<Txn> {
    fn from(host: Host) -> Self {
        Self::Scalar(Scalar::from(host))
    }
}

impl<Txn> From<Link> for State<Txn> {
    fn from(link: Link) -> Self {
        Self::Scalar(Scalar::from(link))
    }
}

impl<Txn> From<Map<InstanceClass>> for State<Txn> {
    fn from(map: Map<InstanceClass>) -> Self {
        Self::Map(
            map.into_iter()
                .map(|(id, class)| (id, State::from(class)))
                .collect(),
        )
    }
}

impl<Txn> From<Map<State<Txn>>> for State<Txn> {
    fn from(map: Map<State<Txn>>) -> Self {
        State::Map(map)
    }
}

impl<Txn> From<Map<Scalar>> for State<Txn> {
    fn from(map: Map<Scalar>) -> Self {
        State::Scalar(map.into())
    }
}

impl<Txn> From<Number> for State<Txn> {
    fn from(n: Number) -> Self {
        Self::Scalar(n.into())
    }
}

impl<Txn> From<Object<Txn>> for State<Txn> {
    fn from(object: Object<Txn>) -> Self {
        State::Object(object)
    }
}

impl<Txn> From<OpDef> for State<Txn> {
    fn from(op_def: OpDef) -> Self {
        State::Scalar(Scalar::Op(op_def))
    }
}

impl<Txn> From<OpRef> for State<Txn> {
    fn from(op_ref: OpRef) -> Self {
        TCRef::Op(op_ref).into()
    }
}

impl<Txn, T> From<Option<T>> for State<Txn>
where
    State<Txn>: From<T>,
{
    fn from(state: Option<T>) -> Self {
        if let Some(state) = state {
            state.into()
        } else {
            Value::None.into()
        }
    }
}

impl<Txn> From<Scalar> for State<Txn> {
    fn from(scalar: Scalar) -> Self {
        State::Scalar(scalar)
    }
}

impl<Txn> From<StateType> for State<Txn> {
    fn from(class: StateType) -> Self {
        Self::Object(InstanceClass::from(class).into())
    }
}

#[cfg(feature = "table")]
impl<Txn> From<Table<Txn>> for State<Txn> {
    fn from(table: Table<Txn>) -> Self {
        Self::Collection(table.into())
    }
}

#[cfg(feature = "tensor")]
impl<Txn> From<Tensor<Txn>> for State<Txn> {
    fn from(tensor: Tensor<Txn>) -> Self {
        Self::Collection(tensor.into())
    }
}

impl<Txn> From<Tuple<State<Txn>>> for State<Txn> {
    fn from(tuple: Tuple<State<Txn>>) -> Self {
        Self::Tuple(tuple)
    }
}

impl<Txn> From<Tuple<Scalar>> for State<Txn> {
    fn from(tuple: Tuple<Scalar>) -> Self {
        Self::Scalar(tuple.into())
    }
}

impl<Txn> From<Tuple<Value>> for State<Txn> {
    fn from(tuple: Tuple<Value>) -> Self {
        Self::Scalar(tuple.into())
    }
}

impl<Txn> From<TCRef> for State<Txn> {
    fn from(tc_ref: TCRef) -> Self {
        Box::new(tc_ref).into()
    }
}

impl<Txn> From<Box<TCRef>> for State<Txn> {
    fn from(tc_ref: Box<TCRef>) -> Self {
        Self::Scalar(Scalar::Ref(tc_ref))
    }
}

impl<Txn> From<Value> for State<Txn> {
    fn from(value: Value) -> Self {
        Self::Scalar(value.into())
    }
}

impl<Txn> From<bool> for State<Txn> {
    fn from(b: bool) -> Self {
        Self::Scalar(b.into())
    }
}

impl<Txn> From<i64> for State<Txn> {
    fn from(n: i64) -> Self {
        Self::Scalar(n.into())
    }
}

impl<Txn> From<usize> for State<Txn> {
    fn from(n: usize) -> Self {
        Self::Scalar(n.into())
    }
}

impl<Txn> From<u64> for State<Txn> {
    fn from(n: u64) -> Self {
        Self::Scalar(n.into())
    }
}

impl<Txn, T1> CastFrom<(T1,)> for State<Txn>
where
    Self: CastFrom<T1>,
{
    fn cast_from(value: (T1,)) -> Self {
        State::Tuple(vec![value.0.cast_into()].into())
    }
}

impl<Txn, T1, T2> CastFrom<(T1, T2)> for State<Txn>
where
    Self: CastFrom<T1>,
    Self: CastFrom<T2>,
{
    fn cast_from(value: (T1, T2)) -> Self {
        State::Tuple(vec![value.0.cast_into(), value.1.cast_into()].into())
    }
}

impl<Txn, T1, T2, T3> CastFrom<(T1, T2, T3)> for State<Txn>
where
    Self: CastFrom<T1>,
    Self: CastFrom<T2>,
    Self: CastFrom<T3>,
{
    fn cast_from(value: (T1, T2, T3)) -> Self {
        State::Tuple(
            vec![
                value.0.cast_into(),
                value.1.cast_into(),
                value.2.cast_into(),
            ]
            .into(),
        )
    }
}

impl<Txn, T1, T2, T3, T4> CastFrom<(T1, T2, T3, T4)> for State<Txn>
where
    Self: CastFrom<T1>,
    Self: CastFrom<T2>,
    Self: CastFrom<T3>,
    Self: CastFrom<T4>,
{
    fn cast_from(value: (T1, T2, T3, T4)) -> Self {
        State::Tuple(
            vec![
                value.0.cast_into(),
                value.1.cast_into(),
                value.2.cast_into(),
                value.3.cast_into(),
            ]
            .into(),
        )
    }
}

impl<Txn> TryFrom<State<Txn>> for bool {
    type Error = TCError;

    fn try_from(state: State<Txn>) -> Result<Self, Self::Error> {
        match state {
            State::Scalar(scalar) => scalar.try_into(),
            other => Err(TCError::unexpected(other, "a boolean")),
        }
    }
}

impl<Txn> TryFrom<State<Txn>> for Id {
    type Error = TCError;

    fn try_from(state: State<Txn>) -> TCResult<Id> {
        match state {
            State::Scalar(scalar) => scalar.try_into(),
            other => Err(TCError::unexpected(other, "an Id")),
        }
    }
}

#[cfg(feature = "collection")]
impl<Txn> TryFrom<State<Txn>> for Collection<Txn> {
    type Error = TCError;

    fn try_from(state: State<Txn>) -> TCResult<Collection<Txn>> {
        match state {
            State::Collection(collection) => Ok(collection),
            other => Err(TCError::unexpected(other, "a Collection")),
        }
    }
}

impl<Txn> TryFrom<State<Txn>> for Scalar {
    type Error = TCError;

    fn try_from(state: State<Txn>) -> TCResult<Self> {
        match state {
            State::Map(map) => map
                .into_iter()
                .map(|(id, state)| Scalar::try_from(state).map(|scalar| (id, scalar)))
                .collect::<TCResult<Map<Scalar>>>()
                .map(Scalar::Map),

            State::Scalar(scalar) => Ok(scalar),

            State::Tuple(tuple) => tuple
                .into_iter()
                .map(|state| Scalar::try_from(state))
                .collect::<TCResult<Tuple<Scalar>>>()
                .map(Scalar::Tuple),

            other => Err(TCError::unexpected(other, "a Scalar")),
        }
    }
}

impl<Txn> TryFrom<State<Txn>> for Map<Scalar> {
    type Error = TCError;

    fn try_from(state: State<Txn>) -> TCResult<Map<Scalar>> {
        match state {
            State::Map(map) => map
                .into_iter()
                .map(|(id, state)| Scalar::try_from(state).map(|scalar| (id, scalar)))
                .collect(),

            State::Scalar(Scalar::Map(map)) => Ok(map),

            State::Tuple(tuple) => tuple
                .into_iter()
                .map(|item| -> TCResult<(Id, Scalar)> { item.try_into() })
                .collect(),

            other => Err(TCError::unexpected(other, "a Map")),
        }
    }
}

impl<Txn> TryFrom<State<Txn>> for Map<State<Txn>> {
    type Error = TCError;

    fn try_from(state: State<Txn>) -> TCResult<Map<State<Txn>>> {
        match state {
            State::Map(map) => Ok(map),

            State::Scalar(Scalar::Map(map)) => Ok(map
                .into_iter()
                .map(|(id, scalar)| (id, State::Scalar(scalar)))
                .collect()),

            State::Tuple(tuple) => tuple
                .into_iter()
                .map(|item| -> TCResult<(Id, State<Txn>)> { item.try_into() })
                .collect(),

            other => Err(TCError::unexpected(other, "a Map")),
        }
    }
}

impl<Txn> TryFrom<State<Txn>> for Map<Value> {
    type Error = TCError;

    fn try_from(state: State<Txn>) -> TCResult<Map<Value>> {
        match state {
            State::Map(map) => map
                .into_iter()
                .map(|(id, state)| Value::try_from(state).map(|scalar| (id, scalar)))
                .collect(),

            State::Scalar(scalar) => scalar.try_into(),

            State::Tuple(tuple) => tuple
                .into_iter()
                .map(|item| -> TCResult<(Id, Value)> { item.try_into() })
                .collect(),

            other => Err(TCError::unexpected(other, "a Map")),
        }
    }
}

impl<Txn> TryFrom<State<Txn>> for Tuple<State<Txn>>
where
    Txn: Send + Sync,
{
    type Error = TCError;

    fn try_from(state: State<Txn>) -> Result<Self, Self::Error> {
        match state {
            State::Map(map) => Ok(map
                .into_iter()
                .map(|(id, state)| State::Tuple(vec![State::from(id), state].into()))
                .collect()),

            State::Scalar(scalar) => {
                let tuple = Tuple::<Scalar>::try_from(scalar)?;
                Ok(tuple.into_iter().map(State::Scalar).collect())
            }

            State::Tuple(tuple) => Ok(tuple),

            other => Err(TCError::unexpected(other, "a Tuple")),
        }
    }
}

impl<Txn, T> TryFrom<State<Txn>> for (Id, T)
where
    T: TryFrom<State<Txn>> + TryFrom<Scalar>,
    TCError: From<<T as TryFrom<State<Txn>>>::Error> + From<<T as TryFrom<Scalar>>::Error>,
{
    type Error = TCError;

    fn try_from(state: State<Txn>) -> TCResult<Self> {
        match state {
            State::Scalar(scalar) => scalar.try_into().map_err(TCError::from),
            State::Tuple(mut tuple) if tuple.len() == 2 => {
                let value = tuple.pop().expect("value");
                let key = tuple.pop().expect("key");
                Ok((key.try_into()?, value.try_into()?))
            }
            other => Err(TCError::unexpected(other, "a map item")),
        }
    }
}

impl<Txn> TryFrom<State<Txn>> for Value {
    type Error = TCError;

    fn try_from(state: State<Txn>) -> TCResult<Value> {
        match state {
            State::Scalar(scalar) => scalar.try_into(),

            State::Tuple(tuple) => tuple
                .into_iter()
                .map(Value::try_from)
                .collect::<TCResult<Tuple<Value>>>()
                .map(Value::Tuple),

            other => Err(TCError::unexpected(other, "a Value")),
        }
    }
}

#[cfg(feature = "chain")]
impl<Txn> TryCastFrom<State<Txn>> for Chain<Txn, CollectionBase<Txn>> {
    fn can_cast_from(state: &State<Txn>) -> bool {
        match state {
            State::Chain(_) => true,
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Chain(chain) => Some(chain),
            _ => None,
        }
    }
}

#[cfg(feature = "chain")]
impl<Txn> TryCastFrom<State<Txn>> for BlockChain<Txn, CollectionBase<Txn>> {
    fn can_cast_from(state: &State<Txn>) -> bool {
        match state {
            State::Chain(Chain::Block(_)) => true,
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Chain(Chain::Block(chain)) => Some(chain),
            _ => None,
        }
    }
}

#[cfg(feature = "chain")]
impl<Txn> TryCastFrom<State<Txn>> for SyncChain<Txn, CollectionBase<Txn>> {
    fn can_cast_from(state: &State<Txn>) -> bool {
        match state {
            State::Chain(Chain::Sync(_)) => true,
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Chain(Chain::Sync(chain)) => Some(chain),
            _ => None,
        }
    }
}

impl<Txn> TryCastFrom<State<Txn>> for Closure<Txn> {
    fn can_cast_from(state: &State<Txn>) -> bool {
        match state {
            State::Closure(_) => true,
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Closure(closure) => Some(closure),
            State::Scalar(scalar) => Self::opt_cast_from(scalar),
            _ => None,
        }
    }
}

impl<Txn> TryCastFrom<State<Txn>> for Box<dyn ClosureInstance<State<Txn>>>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
{
    fn can_cast_from(state: &State<Txn>) -> bool {
        match state {
            State::Closure(_) => true,
            State::Scalar(scalar) => Closure::<Txn>::can_cast_from(scalar),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Closure(closure) => Some(Box::new(closure)),
            State::Scalar(scalar) => {
                if let Some(closure) = Closure::<Txn>::opt_cast_from(scalar) {
                    let closure: Box<dyn ClosureInstance<State<Txn>>> = Box::new(closure);
                    Some(closure)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

#[cfg(feature = "collection")]
impl<Txn> TryCastFrom<State<Txn>> for Collection<Txn> {
    fn can_cast_from(state: &State<Txn>) -> bool {
        match state {
            State::Collection(_) => true,
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Collection(collection) => Some(collection),
            _ => None,
        }
    }
}

#[cfg(feature = "collection")]
impl<Txn> TryCastFrom<State<Txn>> for CollectionBase<Txn> {
    fn can_cast_from(state: &State<Txn>) -> bool {
        match state {
            State::Collection(collection) => CollectionBase::can_cast_from(collection),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Collection(collection) => CollectionBase::opt_cast_from(collection),
            _ => None,
        }
    }
}

#[cfg(any(feature = "btree", feature = "table", feature = "tensor"))]
impl<Txn> TryCastFrom<State<Txn>> for BTree<Txn> {
    fn can_cast_from(state: &State<Txn>) -> bool {
        match state {
            State::Collection(collection) => BTree::can_cast_from(collection),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Collection(collection) => BTree::opt_cast_from(collection),
            _ => None,
        }
    }
}

#[cfg(feature = "table")]
impl<Txn> TryCastFrom<State<Txn>> for Table<Txn> {
    fn can_cast_from(state: &State<Txn>) -> bool {
        match state {
            State::Collection(collection) => Table::can_cast_from(collection),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Collection(collection) => Table::opt_cast_from(collection),
            _ => None,
        }
    }
}

#[cfg(feature = "tensor")]
impl<Txn> TryCastFrom<State<Txn>> for Tensor<Txn> {
    fn can_cast_from(state: &State<Txn>) -> bool {
        match state {
            State::Collection(collection) => Tensor::can_cast_from(collection),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Collection(collection) => Tensor::opt_cast_from(collection),
            _ => None,
        }
    }
}

impl<Txn, T> TryCastFrom<State<Txn>> for (T,)
where
    T: TryCastFrom<State<Txn>>,
{
    fn can_cast_from(state: &State<Txn>) -> bool {
        match state {
            State::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Tuple(tuple) => Self::opt_cast_from(tuple),
            _ => None,
        }
    }
}

impl<Txn, T1, T2> TryCastFrom<State<Txn>> for (T1, T2)
where
    T1: TryCastFrom<State<Txn>>,
    T2: TryCastFrom<State<Txn>>,
{
    fn can_cast_from(state: &State<Txn>) -> bool {
        match state {
            State::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Tuple(tuple) => Self::opt_cast_from(tuple),
            _ => None,
        }
    }
}

impl<Txn, T1, T2, T3> TryCastFrom<State<Txn>> for (T1, T2, T3)
where
    T1: TryCastFrom<State<Txn>>,
    T2: TryCastFrom<State<Txn>>,
    T3: TryCastFrom<State<Txn>>,
{
    fn can_cast_from(state: &State<Txn>) -> bool {
        match state {
            State::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Tuple(tuple) => Self::opt_cast_from(tuple),
            _ => None,
        }
    }
}

// TODO: delete
impl<Txn, T: TryCastFrom<State<Txn>>> TryCastFrom<State<Txn>> for Vec<T> {
    fn can_cast_from(state: &State<Txn>) -> bool {
        match state {
            State::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Tuple(source) => Self::opt_cast_from(source),
            _ => None,
        }
    }
}

// TODO: delete
impl<Txn, T> TryCastFrom<State<Txn>> for Tuple<T>
where
    T: TryCastFrom<State<Txn>>,
{
    fn can_cast_from(state: &State<Txn>) -> bool {
        match state {
            State::Tuple(tuple) => Vec::<T>::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Tuple(tuple) => Vec::<T>::opt_cast_from(tuple).map(Tuple::from),
            _ => None,
        }
    }
}

impl<Txn> TryCastFrom<State<Txn>> for InstanceClass {
    fn can_cast_from(state: &State<Txn>) -> bool {
        match state {
            State::Object(Object::Class(_)) => true,
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            State::Tuple(tuple) => tuple.matches::<(Link, Map<Scalar>)>(),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<InstanceClass> {
        match state {
            State::Object(Object::Class(class)) => Some(class),
            State::Scalar(scalar) => Self::opt_cast_from(scalar),
            State::Tuple(tuple) => {
                let (extends, proto) = tuple.opt_cast_into()?;
                Some(Self::cast_from((extends, proto)))
            }
            _ => None,
        }
    }
}

impl<Txn, T> TryCastFrom<State<Txn>> for Map<T>
where
    T: TryCastFrom<State<Txn>>,
{
    fn can_cast_from(state: &State<Txn>) -> bool {
        match state {
            State::Map(map) => map.values().all(T::can_cast_from),
            State::Tuple(tuple) => tuple.iter().all(|item| item.matches::<(Id, State<Txn>)>()),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Map(map) => {
                let mut dest = Map::new();

                for (key, value) in map {
                    let value = T::opt_cast_from(value)?;
                    dest.insert(key, value);
                }

                Some(dest)
            }
            State::Tuple(tuple) => {
                let mut dest = Map::new();

                for item in tuple {
                    let (key, value): (Id, T) = item.opt_cast_into()?;
                    dest.insert(key, value);
                }

                Some(dest)
            }
            _ => None,
        }
    }
}

impl<Txn> TryCastFrom<State<Txn>> for Scalar {
    fn can_cast_from(state: &State<Txn>) -> bool {
        match state {
            State::Map(map) => map.values().all(Scalar::can_cast_from),
            State::Object(object) => Self::can_cast_from(object),
            State::Scalar(_) => true,
            State::Tuple(tuple) => Vec::<Scalar>::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Map(map) => {
                let mut dest = Map::new();

                for (key, state) in map.into_iter() {
                    let scalar = Scalar::opt_cast_from(state)?;
                    dest.insert(key, scalar);
                }

                Some(Scalar::Map(dest))
            }
            State::Object(object) => Self::opt_cast_from(object),
            State::Scalar(scalar) => Some(scalar),

            State::Tuple(tuple) => {
                let mut scalar = Tuple::<Scalar>::with_capacity(tuple.len());

                for state in tuple {
                    scalar.push(state.opt_cast_into()?);
                }

                Some(Scalar::Tuple(scalar))
            }

            _ => None,
        }
    }
}

impl<Txn> TryCastFrom<State<Txn>> for Value {
    fn can_cast_from(state: &State<Txn>) -> bool {
        match state {
            State::Object(object) => Self::can_cast_from(object),
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            State::Tuple(tuple) => tuple.iter().all(Self::can_cast_from),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Object(object) => Self::opt_cast_from(object),
            State::Scalar(scalar) => Self::opt_cast_from(scalar),

            State::Tuple(tuple) => {
                let mut value = Tuple::<Value>::with_capacity(tuple.len());

                for state in tuple {
                    value.push(state.opt_cast_into()?);
                }

                Some(Value::Tuple(value))
            }

            _ => None,
        }
    }
}

impl<Txn> TryCastFrom<State<Txn>> for Link {
    fn can_cast_from(state: &State<Txn>) -> bool {
        match state {
            State::Object(Object::Class(class)) => Self::can_cast_from(class),
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn>) -> Option<Self> {
        match state {
            State::Object(Object::Class(class)) => Self::opt_cast_from(class).map(Self::from),
            State::Scalar(scalar) => Self::opt_cast_from(scalar).map(Self::from),
            _ => None,
        }
    }
}

macro_rules! from_scalar {
    ($t:ty) => {
        impl<Txn> TryCastFrom<State<Txn>> for $t {
            fn can_cast_from(state: &State<Txn>) -> bool {
                match state {
                    State::Scalar(scalar) => Self::can_cast_from(scalar),
                    _ => false,
                }
            }

            fn opt_cast_from(state: State<Txn>) -> Option<Self> {
                match state {
                    State::Scalar(scalar) => Self::opt_cast_from(scalar),
                    _ => None,
                }
            }
        }
    };
}

from_scalar!(Bytes);
from_scalar!(Float);
from_scalar!(Id);
from_scalar!(IdRef);
from_scalar!(Host);
from_scalar!(Number);
from_scalar!(OpDef);
from_scalar!(OpRef);
from_scalar!(TCPathBuf);
from_scalar!(TCString);
from_scalar!(bool);
from_scalar!(usize);
from_scalar!(u64);

impl<Txn> fmt::Debug for State<Txn> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            #[cfg(feature = "chain")]
            Self::Chain(chain) => fmt::Debug::fmt(chain, f),
            Self::Closure(closure) => fmt::Debug::fmt(closure, f),
            #[cfg(feature = "collection")]
            Self::Collection(collection) => fmt::Debug::fmt(collection, f),
            Self::Map(map) => fmt::Debug::fmt(map, f),
            Self::Object(object) => fmt::Debug::fmt(object, f),
            Self::Scalar(scalar) => fmt::Debug::fmt(scalar, f),
            Self::Tuple(tuple) => fmt::Debug::fmt(tuple, f),
        }
    }
}

struct StateVisitor<Txn> {
    txn: Txn,
    scalar: ScalarVisitor,
}

impl<Txn> StateVisitor<Txn>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
{
    async fn visit_map_value<A: de::MapAccess>(
        &self,
        class: StateType,
        access: &mut A,
    ) -> Result<State<Txn>, A::Error> {
        debug!("decode instance of {:?}", class);

        match class {
            #[cfg(feature = "chain")]
            StateType::Chain(ct) => {
                ChainVisitor::new(self.txn.clone())
                    .visit_map_value(ct, access)
                    .map_ok(State::Chain)
                    .await
            }
            StateType::Closure => {
                access
                    .next_value(self.txn.clone())
                    .map_ok(State::Closure)
                    .await
            }
            #[cfg(feature = "collection")]
            StateType::Collection(ct) => {
                CollectionVisitor::new(self.txn.clone())
                    .visit_map_value(ct, access)
                    .map_ok(Collection::from)
                    .map_ok(State::Collection)
                    .await
            }
            StateType::Map => access.next_value(self.txn.clone()).await,
            StateType::Object(ot) => {
                let txn = self.txn.subcontext_unique();

                let state = access.next_value(txn).await?;
                ObjectVisitor::new()
                    .visit_map_value(ot, state)
                    .map_ok(State::Object)
                    .await
            }
            StateType::Scalar(st) => {
                ScalarVisitor::visit_map_value(st, access)
                    .map_ok(State::Scalar)
                    .await
            }
            StateType::Tuple => access.next_value(self.txn.clone()).await,
        }
    }
}

#[async_trait]
impl<'a, Txn> de::Visitor for StateVisitor<Txn>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
{
    type Value = State<Txn>;

    fn expecting() -> &'static str {
        r#"a State, e.g. 1 or [2] or "three" or {"/state/scalar/value/number/complex": [3.14, -1.414]}"#
    }

    async fn visit_array_u8<A: ArrayAccess<u8>>(self, array: A) -> Result<Self::Value, A::Error> {
        self.scalar
            .visit_array_u8(array)
            .map_ok(State::Scalar)
            .await
    }

    fn visit_bool<E: de::Error>(self, b: bool) -> Result<Self::Value, E> {
        self.scalar.visit_bool(b).map(State::Scalar)
    }

    fn visit_i8<E: de::Error>(self, i: i8) -> Result<Self::Value, E> {
        self.scalar.visit_i8(i).map(State::Scalar)
    }

    fn visit_i16<E: de::Error>(self, i: i16) -> Result<Self::Value, E> {
        self.scalar.visit_i16(i).map(State::Scalar)
    }

    fn visit_i32<E: de::Error>(self, i: i32) -> Result<Self::Value, E> {
        self.scalar.visit_i32(i).map(State::Scalar)
    }

    fn visit_i64<E: de::Error>(self, i: i64) -> Result<Self::Value, E> {
        self.scalar.visit_i64(i).map(State::Scalar)
    }

    fn visit_u8<E: de::Error>(self, u: u8) -> Result<Self::Value, E> {
        self.scalar.visit_u8(u).map(State::Scalar)
    }

    fn visit_u16<E: de::Error>(self, u: u16) -> Result<Self::Value, E> {
        self.scalar.visit_u16(u).map(State::Scalar)
    }

    fn visit_u32<E: de::Error>(self, u: u32) -> Result<Self::Value, E> {
        self.scalar.visit_u32(u).map(State::Scalar)
    }

    fn visit_u64<E: de::Error>(self, u: u64) -> Result<Self::Value, E> {
        self.scalar.visit_u64(u).map(State::Scalar)
    }

    fn visit_f32<E: de::Error>(self, f: f32) -> Result<Self::Value, E> {
        self.scalar.visit_f32(f).map(State::Scalar)
    }

    fn visit_f64<E: de::Error>(self, f: f64) -> Result<Self::Value, E> {
        self.scalar.visit_f64(f).map(State::Scalar)
    }

    fn visit_string<E: de::Error>(self, s: String) -> Result<Self::Value, E> {
        self.scalar.visit_string(s).map(State::Scalar)
    }

    fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
        self.scalar.visit_unit().map(State::Scalar)
    }

    fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
        self.scalar.visit_none().map(State::Scalar)
    }

    async fn visit_map<A: de::MapAccess>(self, mut access: A) -> Result<Self::Value, A::Error> {
        if let Some(key) = access.next_key::<String>(()).await? {
            debug!("deserialize: key is {}", key);

            if key.starts_with('/') {
                if let Ok(path) = TCPathBuf::from_str(&key) {
                    debug!("is {} a classpath?", path);

                    if let Some(class) = StateType::from_path(&path) {
                        debug!("deserialize instance of {:?}...", class);
                        return self.visit_map_value(class, &mut access).await;
                    } else {
                        debug!("not a classpath: {}", path);
                    }
                }
            }

            if let Ok(subject) = reference::Subject::from_str(&key) {
                let params = access.next_value(()).await?;
                debug!("deserialize Scalar from key {} and value {:?}", key, params);
                return ScalarVisitor::visit_subject(subject, params).map(State::Scalar);
            }

            let mut map = Map::<State<Txn>>::new();

            let id = Id::from_str(&key).map_err(de::Error::custom)?;
            let txn = self.txn.subcontext(id.clone());
            let value = access.next_value(txn).await?;
            map.insert(id.clone(), value);

            while let Some(id) = access.next_key::<Id>(()).await? {
                let txn = self.txn.subcontext(id.clone());
                let state = access.next_value(txn).await?;
                map.insert(id, state);
            }

            debug!("deserialize map {map:?}");
            if map.len() == 1 {
                let value = map.as_ref().get(&id).expect("map value");
                if value.is_none() && !value.is_map() {
                    return Ok(State::from(id));
                }
            }

            Ok(State::Map(map))
        } else {
            Ok(State::Map(Map::default()))
        }
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut access: A) -> Result<Self::Value, A::Error> {
        let mut seq = if let Some(len) = access.size_hint() {
            Vec::with_capacity(len)
        } else {
            Vec::new()
        };

        let mut i = 0usize;
        loop {
            let txn = self.txn.subcontext(i);

            if let Some(next) = access.next_element(txn).await? {
                seq.push(next);
                i += 1;
            } else {
                break;
            }
        }

        Ok(State::Tuple(seq.into()))
    }
}

#[async_trait]
impl<Txn> de::FromStream for State<Txn>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
{
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        let scalar = ScalarVisitor::default();
        decoder.decode_any(StateVisitor { txn, scalar }).await
    }
}
