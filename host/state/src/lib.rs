//! A TinyChain [`State`]

use std::collections::HashSet;
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::marker::PhantomData;
use std::str::FromStr;

use async_hash::{Digest, Hash, Output, Sha256};
use async_trait::async_trait;
use bytes::Bytes;
use destream::de;
use destream::ArrayAccess;
use futures::future::TryFutureExt;
use futures::stream::{self, StreamExt, TryStreamExt};
use log::debug;
use safecast::*;

use tc_chain::{ChainType, ChainVisitor};
use tc_collection::btree::BTreeType;
use tc_collection::table::TableType;
use tc_collection::tensor::TensorType;
use tc_collection::{CollectionType, CollectionVisitor};
use tc_error::*;
use tc_scalar::*;
use tc_transact::public::{ClosureInstance, Public, StateInstance, ToState};
use tc_transact::{fs, AsyncHash, RPCClient, Transaction, TxnId};
use tc_value::{Float, Host, Link, Number, NumberType, TCString, Value, ValueType};
use tcgeneric::{
    path_label, Class, Id, Instance, Map, NativeClass, PathSegment, TCPath, TCPathBuf, Tuple,
};

use chain::*;
use closure::*;
use collection::*;
use object::{InstanceClass, Object, ObjectType, ObjectVisitor};

pub use tc_chain as chain;
pub use tc_collection as collection;
use tc_transact::fs::FileSave;

pub mod closure;
pub mod object;
pub mod public;
pub mod view;

/// The [`Class`] of a [`State`].
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum StateType {
    Chain(ChainType),
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
                    "collection" => CollectionType::from_path(path).map(Self::Collection),
                    "chain" => ChainType::from_path(path).map(Self::Chain),
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
            Self::Collection(ct) => ct.path(),
            Self::Chain(ct) => ct.path(),
            Self::Closure => path_label(&["state", "closure"]).into(),
            Self::Map => path_label(&["state", "map"]).into(),
            Self::Object(ot) => ot.path(),
            Self::Scalar(st) => st.path(),
            Self::Tuple => path_label(&["state", "tuple"]).into(),
        }
    }
}

impl From<BTreeType> for StateType {
    fn from(btt: BTreeType) -> Self {
        CollectionType::BTree(btt).into()
    }
}

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

impl From<TableType> for StateType {
    fn from(tt: TableType) -> Self {
        Self::Collection(tt.into())
    }
}

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
            Self::Chain(ct) => fmt::Debug::fmt(ct, f),
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
pub enum State<Txn, FE> {
    Collection(Collection<Txn, FE>),
    Chain(Chain<Self, Txn, FE, CollectionBase<Txn, FE>>),
    Closure(Closure<Txn, FE>),
    Map(Map<Self>),
    Object(Object<Txn, FE>),
    Scalar(Scalar),
    Tuple(Tuple<Self>),
}

impl<Txn, FE> Clone for State<Txn, FE> {
    fn clone(&self) -> Self {
        match self {
            Self::Collection(collection) => Self::Collection(collection.clone()),
            Self::Chain(chain) => Self::Chain(chain.clone()),
            Self::Closure(closure) => Self::Closure(closure.clone()),
            Self::Map(map) => Self::Map(map.clone()),
            Self::Object(obj) => Self::Object(obj.clone()),
            Self::Scalar(scalar) => Self::Scalar(scalar.clone()),
            Self::Tuple(tuple) => Self::Tuple(tuple.clone()),
        }
    }
}

impl<Txn, FE> State<Txn, FE> {
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

impl<Txn, FE> State<Txn, FE>
where
    Txn: Transaction<FE> + RPCClient<Self>,
    FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'a> fs::FileSave<'a>
        + Clone,
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

impl<Txn, FE> StateInstance for State<Txn, FE>
where
    Txn: Transaction<FE> + RPCClient<Self>,
    FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'a> fs::FileSave<'a>
        + Clone,
{
    type FE = FE;
    type Txn = Txn;
    type Closure = Closure<Txn, FE>;

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
impl<Txn, FE> Refer<Self> for State<Txn, FE>
where
    Txn: Transaction<FE> + RPCClient<Self>,
    FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'a> fs::FileSave<'a>
        + Clone,
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

impl<Txn, FE> Default for State<Txn, FE> {
    fn default() -> Self {
        Self::Scalar(Scalar::default())
    }
}

impl<Txn, FE> Instance for State<Txn, FE>
where
    Txn: Send + Sync,
    FE: Send + Sync,
{
    type Class = StateType;

    fn class(&self) -> StateType {
        match self {
            Self::Chain(chain) => StateType::Chain(chain.class()),
            Self::Closure(_) => StateType::Closure,
            Self::Collection(collection) => StateType::Collection(collection.class()),
            Self::Map(_) => StateType::Map,
            Self::Object(object) => StateType::Object(object.class()),
            Self::Scalar(scalar) => StateType::Scalar(scalar.class()),
            Self::Tuple(_) => StateType::Tuple,
        }
    }
}

#[async_trait]
impl<Txn, FE> AsyncHash for State<Txn, FE>
where
    Txn: Transaction<FE> + RPCClient<Self>,
    FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<TensorNode>
        + AsType<ChainBlock>
        + for<'a> fs::FileSave<'a>
        + Clone,
{
    async fn hash(self, txn_id: TxnId) -> TCResult<Output<Sha256>> {
        match self {
            Self::Collection(collection) => collection.hash(txn_id).await,
            Self::Chain(chain) => chain.hash(txn_id).await,
            Self::Closure(closure) => closure.hash(txn_id).await,
            Self::Map(map) => {
                let mut hashes = stream::iter(map)
                    .map(|(id, state)| {
                        state
                            .hash(txn_id)
                            .map_ok(|hash| (Hash::<Sha256>::hash(id), hash))
                    })
                    .buffered(num_cpus::get())
                    .map_ok(|(id, state)| {
                        let mut inner_hasher = Sha256::default();
                        inner_hasher.update(&id);
                        inner_hasher.update(&state);
                        inner_hasher.finalize()
                    });

                let mut hasher = Sha256::default();
                while let Some(hash) = hashes.try_next().await? {
                    hasher.update(&hash);
                }

                Ok(hasher.finalize())
            }
            Self::Object(object) => object.hash(txn_id).await,
            Self::Scalar(scalar) => Ok(Hash::<Sha256>::hash(scalar)),
            Self::Tuple(tuple) => {
                let mut hashes = stream::iter(tuple)
                    .map(|state| state.hash(txn_id))
                    .buffered(num_cpus::get());

                let mut hasher = Sha256::default();
                while let Some(hash) = hashes.try_next().await? {
                    hasher.update(&hash);
                }
                Ok(hasher.finalize())
            }
        }
    }
}

impl<Txn, FE> From<()> for State<Txn, FE> {
    fn from(_: ()) -> Self {
        State::Scalar(Scalar::Value(Value::None))
    }
}

impl<Txn, FE> From<BTree<Txn, FE>> for State<Txn, FE> {
    fn from(btree: BTree<Txn, FE>) -> Self {
        Self::Collection(btree.into())
    }
}

impl<Txn, FE> From<Chain<Self, Txn, FE, CollectionBase<Txn, FE>>> for State<Txn, FE> {
    fn from(chain: Chain<Self, Txn, FE, CollectionBase<Txn, FE>>) -> Self {
        Self::Chain(chain)
    }
}

impl<Txn, FE> From<BlockChain<Self, Txn, FE, CollectionBase<Txn, FE>>> for State<Txn, FE> {
    fn from(chain: BlockChain<Self, Txn, FE, CollectionBase<Txn, FE>>) -> Self {
        Self::Chain(chain.into())
    }
}

impl<Txn, FE> From<Closure<Txn, FE>> for State<Txn, FE> {
    fn from(closure: Closure<Txn, FE>) -> Self {
        Self::Closure(closure)
    }
}

impl<Txn, FE> From<Collection<Txn, FE>> for State<Txn, FE> {
    fn from(collection: Collection<Txn, FE>) -> Self {
        Self::Collection(collection)
    }
}

impl<Txn, FE> From<CollectionBase<Txn, FE>> for State<Txn, FE> {
    fn from(collection: CollectionBase<Txn, FE>) -> Self {
        Self::Collection(collection.into())
    }
}

impl<Txn, FE> From<Id> for State<Txn, FE> {
    fn from(id: Id) -> Self {
        Self::Scalar(id.into())
    }
}

impl<Txn, FE> From<InstanceClass> for State<Txn, FE> {
    fn from(class: InstanceClass) -> Self {
        Self::Object(class.into())
    }
}

impl<Txn, FE> From<Host> for State<Txn, FE> {
    fn from(host: Host) -> Self {
        Self::Scalar(Scalar::from(host))
    }
}

impl<Txn, FE> From<Link> for State<Txn, FE> {
    fn from(link: Link) -> Self {
        Self::Scalar(Scalar::from(link))
    }
}

impl<Txn, FE> From<Map<InstanceClass>> for State<Txn, FE> {
    fn from(map: Map<InstanceClass>) -> Self {
        Self::Map(
            map.into_iter()
                .map(|(id, class)| (id, State::from(class)))
                .collect(),
        )
    }
}

impl<Txn, FE> From<Map<State<Txn, FE>>> for State<Txn, FE> {
    fn from(map: Map<State<Txn, FE>>) -> Self {
        State::Map(map)
    }
}

impl<Txn, FE> From<Map<Scalar>> for State<Txn, FE> {
    fn from(map: Map<Scalar>) -> Self {
        State::Scalar(map.into())
    }
}

impl<Txn, FE> From<Number> for State<Txn, FE> {
    fn from(n: Number) -> Self {
        Self::Scalar(n.into())
    }
}

impl<Txn, FE> From<Object<Txn, FE>> for State<Txn, FE> {
    fn from(object: Object<Txn, FE>) -> Self {
        State::Object(object)
    }
}

impl<Txn, FE> From<OpDef> for State<Txn, FE> {
    fn from(op_def: OpDef) -> Self {
        State::Scalar(Scalar::Op(op_def))
    }
}

impl<Txn, FE> From<OpRef> for State<Txn, FE> {
    fn from(op_ref: OpRef) -> Self {
        TCRef::Op(op_ref).into()
    }
}

impl<Txn, FE, T> From<Option<T>> for State<Txn, FE>
where
    State<Txn, FE>: From<T>,
{
    fn from(state: Option<T>) -> Self {
        if let Some(state) = state {
            state.into()
        } else {
            Value::None.into()
        }
    }
}

impl<Txn, FE> From<Scalar> for State<Txn, FE> {
    fn from(scalar: Scalar) -> Self {
        State::Scalar(scalar)
    }
}

impl<Txn, FE> From<StateType> for State<Txn, FE> {
    fn from(class: StateType) -> Self {
        Self::Object(InstanceClass::from(class).into())
    }
}

impl<Txn, FE> From<Table<Txn, FE>> for State<Txn, FE> {
    fn from(table: Table<Txn, FE>) -> Self {
        Self::Collection(table.into())
    }
}

impl<Txn, FE> From<Tensor<Txn, FE>> for State<Txn, FE> {
    fn from(tensor: Tensor<Txn, FE>) -> Self {
        Self::Collection(tensor.into())
    }
}

impl<Txn, FE> From<Tuple<State<Txn, FE>>> for State<Txn, FE> {
    fn from(tuple: Tuple<State<Txn, FE>>) -> Self {
        Self::Tuple(tuple)
    }
}

impl<Txn, FE> From<Tuple<Scalar>> for State<Txn, FE> {
    fn from(tuple: Tuple<Scalar>) -> Self {
        Self::Scalar(tuple.into())
    }
}

impl<Txn, FE> From<Tuple<Value>> for State<Txn, FE> {
    fn from(tuple: Tuple<Value>) -> Self {
        Self::Scalar(tuple.into())
    }
}

impl<Txn, FE> From<TCRef> for State<Txn, FE> {
    fn from(tc_ref: TCRef) -> Self {
        Box::new(tc_ref).into()
    }
}

impl<Txn, FE> From<Box<TCRef>> for State<Txn, FE> {
    fn from(tc_ref: Box<TCRef>) -> Self {
        Self::Scalar(Scalar::Ref(tc_ref))
    }
}

impl<Txn, FE> From<Value> for State<Txn, FE> {
    fn from(value: Value) -> Self {
        Self::Scalar(value.into())
    }
}

impl<Txn, FE> From<bool> for State<Txn, FE> {
    fn from(b: bool) -> Self {
        Self::Scalar(b.into())
    }
}

impl<Txn, FE> From<i64> for State<Txn, FE> {
    fn from(n: i64) -> Self {
        Self::Scalar(n.into())
    }
}

impl<Txn, FE> From<usize> for State<Txn, FE> {
    fn from(n: usize) -> Self {
        Self::Scalar(n.into())
    }
}

impl<Txn, FE> From<u64> for State<Txn, FE> {
    fn from(n: u64) -> Self {
        Self::Scalar(n.into())
    }
}

impl<Txn, FE, T1> CastFrom<(T1,)> for State<Txn, FE>
where
    Self: CastFrom<T1>,
{
    fn cast_from(value: (T1,)) -> Self {
        State::Tuple(vec![value.0.cast_into()].into())
    }
}

impl<Txn, FE, T1, T2> CastFrom<(T1, T2)> for State<Txn, FE>
where
    Self: CastFrom<T1>,
    Self: CastFrom<T2>,
{
    fn cast_from(value: (T1, T2)) -> Self {
        State::Tuple(vec![value.0.cast_into(), value.1.cast_into()].into())
    }
}

impl<Txn, FE, T1, T2, T3> CastFrom<(T1, T2, T3)> for State<Txn, FE>
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

impl<Txn, FE, T1, T2, T3, T4> CastFrom<(T1, T2, T3, T4)> for State<Txn, FE>
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

impl<Txn, FE> TryFrom<State<Txn, FE>> for bool {
    type Error = TCError;

    fn try_from(state: State<Txn, FE>) -> Result<Self, Self::Error> {
        match state {
            State::Scalar(scalar) => scalar.try_into(),
            other => Err(TCError::unexpected(other, "a boolean")),
        }
    }
}

impl<Txn, FE> TryFrom<State<Txn, FE>> for Id {
    type Error = TCError;

    fn try_from(state: State<Txn, FE>) -> TCResult<Id> {
        match state {
            State::Scalar(scalar) => scalar.try_into(),
            other => Err(TCError::unexpected(other, "an Id")),
        }
    }
}

impl<Txn, FE> TryFrom<State<Txn, FE>> for Collection<Txn, FE> {
    type Error = TCError;

    fn try_from(state: State<Txn, FE>) -> TCResult<Collection<Txn, FE>> {
        match state {
            State::Collection(collection) => Ok(collection),
            other => Err(TCError::unexpected(other, "a Collection")),
        }
    }
}

impl<Txn, FE> TryFrom<State<Txn, FE>> for Scalar {
    type Error = TCError;

    fn try_from(state: State<Txn, FE>) -> TCResult<Self> {
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

impl<Txn, FE> TryFrom<State<Txn, FE>> for Map<Scalar> {
    type Error = TCError;

    fn try_from(state: State<Txn, FE>) -> TCResult<Map<Scalar>> {
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

impl<Txn, FE> TryFrom<State<Txn, FE>> for Map<State<Txn, FE>> {
    type Error = TCError;

    fn try_from(state: State<Txn, FE>) -> TCResult<Map<State<Txn, FE>>> {
        match state {
            State::Map(map) => Ok(map),

            State::Scalar(Scalar::Map(map)) => Ok(map
                .into_iter()
                .map(|(id, scalar)| (id, State::Scalar(scalar)))
                .collect()),

            State::Tuple(tuple) => tuple
                .into_iter()
                .map(|item| -> TCResult<(Id, State<Txn, FE>)> { item.try_into() })
                .collect(),

            other => Err(TCError::unexpected(other, "a Map")),
        }
    }
}

impl<Txn, FE> TryFrom<State<Txn, FE>> for Map<Value> {
    type Error = TCError;

    fn try_from(state: State<Txn, FE>) -> TCResult<Map<Value>> {
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

impl<Txn, FE> TryFrom<State<Txn, FE>> for Tuple<State<Txn, FE>>
where
    Txn: Send + Sync,
    FE: Send + Sync,
{
    type Error = TCError;

    fn try_from(state: State<Txn, FE>) -> Result<Self, Self::Error> {
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

impl<Txn, FE, T> TryFrom<State<Txn, FE>> for (Id, T)
where
    T: TryFrom<State<Txn, FE>> + TryFrom<Scalar>,
    TCError: From<<T as TryFrom<State<Txn, FE>>>::Error> + From<<T as TryFrom<Scalar>>::Error>,
{
    type Error = TCError;

    fn try_from(state: State<Txn, FE>) -> TCResult<Self> {
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

impl<Txn, FE> TryFrom<State<Txn, FE>> for Value {
    type Error = TCError;

    fn try_from(state: State<Txn, FE>) -> TCResult<Value> {
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

impl<Txn, FE> TryCastFrom<State<Txn, FE>>
    for Chain<State<Txn, FE>, Txn, FE, CollectionBase<Txn, FE>>
{
    fn can_cast_from(state: &State<Txn, FE>) -> bool {
        match state {
            State::Chain(_) => true,
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn, FE>) -> Option<Self> {
        match state {
            State::Chain(chain) => Some(chain),
            _ => None,
        }
    }
}

impl<Txn, FE> TryCastFrom<State<Txn, FE>>
    for BlockChain<State<Txn, FE>, Txn, FE, CollectionBase<Txn, FE>>
{
    fn can_cast_from(state: &State<Txn, FE>) -> bool {
        match state {
            State::Chain(Chain::Block(_)) => true,
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn, FE>) -> Option<Self> {
        match state {
            State::Chain(Chain::Block(chain)) => Some(chain),
            _ => None,
        }
    }
}

impl<Txn, FE> TryCastFrom<State<Txn, FE>> for Closure<Txn, FE> {
    fn can_cast_from(state: &State<Txn, FE>) -> bool {
        match state {
            State::Closure(_) => true,
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn, FE>) -> Option<Self> {
        match state {
            State::Closure(closure) => Some(closure),
            State::Scalar(scalar) => Self::opt_cast_from(scalar),
            _ => None,
        }
    }
}

impl<Txn, FE> TryCastFrom<State<Txn, FE>> for Box<dyn ClosureInstance<State<Txn, FE>>>
where
    Txn: Transaction<FE> + RPCClient<State<Txn, FE>>,
    FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'a> fs::FileSave<'a>
        + Clone,
{
    fn can_cast_from(state: &State<Txn, FE>) -> bool {
        match state {
            State::Closure(_) => true,
            State::Scalar(scalar) => Closure::<Txn, FE>::can_cast_from(scalar),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn, FE>) -> Option<Self> {
        match state {
            State::Closure(closure) => Some(Box::new(closure)),
            State::Scalar(scalar) => {
                if let Some(closure) = Closure::<Txn, FE>::opt_cast_from(scalar) {
                    let closure: Box<dyn ClosureInstance<State<Txn, FE>>> = Box::new(closure);
                    Some(closure)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

impl<Txn, FE> TryCastFrom<State<Txn, FE>> for Collection<Txn, FE> {
    fn can_cast_from(state: &State<Txn, FE>) -> bool {
        match state {
            State::Collection(_) => true,
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn, FE>) -> Option<Self> {
        match state {
            State::Collection(collection) => Some(collection),
            _ => None,
        }
    }
}

impl<Txn, FE> TryCastFrom<State<Txn, FE>> for CollectionBase<Txn, FE> {
    fn can_cast_from(state: &State<Txn, FE>) -> bool {
        match state {
            State::Collection(collection) => CollectionBase::can_cast_from(collection),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn, FE>) -> Option<Self> {
        match state {
            State::Collection(collection) => CollectionBase::opt_cast_from(collection),
            _ => None,
        }
    }
}

impl<Txn, FE> TryCastFrom<State<Txn, FE>> for BTree<Txn, FE> {
    fn can_cast_from(state: &State<Txn, FE>) -> bool {
        match state {
            State::Collection(collection) => BTree::can_cast_from(collection),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn, FE>) -> Option<Self> {
        match state {
            State::Collection(collection) => BTree::opt_cast_from(collection),
            _ => None,
        }
    }
}

impl<Txn, FE> TryCastFrom<State<Txn, FE>> for Table<Txn, FE> {
    fn can_cast_from(state: &State<Txn, FE>) -> bool {
        match state {
            State::Collection(collection) => Table::can_cast_from(collection),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn, FE>) -> Option<Self> {
        match state {
            State::Collection(collection) => Table::opt_cast_from(collection),
            _ => None,
        }
    }
}

impl<Txn, FE> TryCastFrom<State<Txn, FE>> for Tensor<Txn, FE> {
    fn can_cast_from(state: &State<Txn, FE>) -> bool {
        match state {
            State::Collection(collection) => Tensor::can_cast_from(collection),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn, FE>) -> Option<Self> {
        match state {
            State::Collection(collection) => Tensor::opt_cast_from(collection),
            _ => None,
        }
    }
}

impl<Txn, FE, T> TryCastFrom<State<Txn, FE>> for (T,)
where
    T: TryCastFrom<State<Txn, FE>>,
{
    fn can_cast_from(state: &State<Txn, FE>) -> bool {
        match state {
            State::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn, FE>) -> Option<Self> {
        match state {
            State::Tuple(tuple) => Self::opt_cast_from(tuple),
            _ => None,
        }
    }
}

impl<Txn, FE, T1, T2> TryCastFrom<State<Txn, FE>> for (T1, T2)
where
    T1: TryCastFrom<State<Txn, FE>>,
    T2: TryCastFrom<State<Txn, FE>>,
{
    fn can_cast_from(state: &State<Txn, FE>) -> bool {
        match state {
            State::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn, FE>) -> Option<Self> {
        match state {
            State::Tuple(tuple) => Self::opt_cast_from(tuple),
            _ => None,
        }
    }
}

impl<Txn, FE, T1, T2, T3> TryCastFrom<State<Txn, FE>> for (T1, T2, T3)
where
    T1: TryCastFrom<State<Txn, FE>>,
    T2: TryCastFrom<State<Txn, FE>>,
    T3: TryCastFrom<State<Txn, FE>>,
{
    fn can_cast_from(state: &State<Txn, FE>) -> bool {
        match state {
            State::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn, FE>) -> Option<Self> {
        match state {
            State::Tuple(tuple) => Self::opt_cast_from(tuple),
            _ => None,
        }
    }
}

// TODO: delete
impl<Txn, FE, T: TryCastFrom<State<Txn, FE>>> TryCastFrom<State<Txn, FE>> for Vec<T> {
    fn can_cast_from(state: &State<Txn, FE>) -> bool {
        match state {
            State::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn, FE>) -> Option<Self> {
        match state {
            State::Tuple(source) => Self::opt_cast_from(source),
            _ => None,
        }
    }
}

// TODO: delete
impl<Txn, FE, T> TryCastFrom<State<Txn, FE>> for Tuple<T>
where
    T: TryCastFrom<State<Txn, FE>>,
{
    fn can_cast_from(state: &State<Txn, FE>) -> bool {
        match state {
            State::Tuple(tuple) => Vec::<T>::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn, FE>) -> Option<Self> {
        match state {
            State::Tuple(tuple) => Vec::<T>::opt_cast_from(tuple).map(Tuple::from),
            _ => None,
        }
    }
}

impl<Txn, FE> TryCastFrom<State<Txn, FE>> for InstanceClass {
    fn can_cast_from(state: &State<Txn, FE>) -> bool {
        match state {
            State::Object(Object::Class(_)) => true,
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            State::Tuple(tuple) => tuple.matches::<(Link, Map<Scalar>)>(),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn, FE>) -> Option<InstanceClass> {
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

impl<Txn, FE, T> TryCastFrom<State<Txn, FE>> for Map<T>
where
    T: TryCastFrom<State<Txn, FE>>,
{
    fn can_cast_from(state: &State<Txn, FE>) -> bool {
        match state {
            State::Map(map) => map.values().all(T::can_cast_from),
            State::Tuple(tuple) => tuple
                .iter()
                .all(|item| item.matches::<(Id, State<Txn, FE>)>()),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn, FE>) -> Option<Self> {
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

impl<Txn, FE> TryCastFrom<State<Txn, FE>> for Scalar {
    fn can_cast_from(state: &State<Txn, FE>) -> bool {
        match state {
            State::Map(map) => map.values().all(Scalar::can_cast_from),
            State::Object(object) => Self::can_cast_from(object),
            State::Scalar(_) => true,
            State::Tuple(tuple) => Vec::<Scalar>::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn, FE>) -> Option<Self> {
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

impl<Txn, FE> TryCastFrom<State<Txn, FE>> for Value {
    fn can_cast_from(state: &State<Txn, FE>) -> bool {
        match state {
            State::Object(object) => Self::can_cast_from(object),
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            State::Tuple(tuple) => tuple.iter().all(Self::can_cast_from),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn, FE>) -> Option<Self> {
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

impl<Txn, FE> TryCastFrom<State<Txn, FE>> for Link {
    fn can_cast_from(state: &State<Txn, FE>) -> bool {
        match state {
            State::Object(Object::Class(class)) => Self::can_cast_from(class),
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            _ => false,
        }
    }

    fn opt_cast_from(state: State<Txn, FE>) -> Option<Self> {
        match state {
            State::Object(Object::Class(class)) => Self::opt_cast_from(class).map(Self::from),
            State::Scalar(scalar) => Self::opt_cast_from(scalar).map(Self::from),
            _ => None,
        }
    }
}

macro_rules! from_scalar {
    ($t:ty) => {
        impl<Txn, FE> TryCastFrom<State<Txn, FE>> for $t {
            fn can_cast_from(state: &State<Txn, FE>) -> bool {
                match state {
                    State::Scalar(scalar) => Self::can_cast_from(scalar),
                    _ => false,
                }
            }

            fn opt_cast_from(state: State<Txn, FE>) -> Option<Self> {
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

impl<Txn, FE> fmt::Debug for State<Txn, FE> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Chain(chain) => fmt::Debug::fmt(chain, f),
            Self::Closure(closure) => fmt::Debug::fmt(closure, f),
            Self::Collection(collection) => fmt::Debug::fmt(collection, f),
            Self::Map(map) => fmt::Debug::fmt(map, f),
            Self::Object(object) => fmt::Debug::fmt(object, f),
            Self::Scalar(scalar) => fmt::Debug::fmt(scalar, f),
            Self::Tuple(tuple) => fmt::Debug::fmt(tuple, f),
        }
    }
}

struct StateVisitor<Txn, FE> {
    txn: Txn,
    scalar: ScalarVisitor,
    phantom: PhantomData<FE>,
}

impl<Txn, FE> StateVisitor<Txn, FE>
where
    Txn: Transaction<FE> + RPCClient<State<Txn, FE>>,
    FE: AsType<BTreeNode>
        + AsType<TensorNode>
        + AsType<ChainBlock>
        + DenseCacheFile
        + for<'a> FileSave<'a>
        + Clone,
{
    async fn visit_map_value<A: de::MapAccess>(
        &self,
        class: StateType,
        access: &mut A,
    ) -> Result<State<Txn, FE>, A::Error> {
        debug!("decode instance of {:?}", class);

        match class {
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

// TODO: guard against a DoS attack using an infinite request stream
#[async_trait]
impl<'a, Txn, FE> de::Visitor for StateVisitor<Txn, FE>
where
    Txn: Transaction<FE> + RPCClient<State<Txn, FE>>,
    FE: AsType<BTreeNode>
        + AsType<TensorNode>
        + AsType<ChainBlock>
        + DenseCacheFile
        + for<'b> FileSave<'b>
        + Clone,
{
    type Value = State<Txn, FE>;

    fn expecting() -> &'static str {
        "a State, e.g. 1 or [2] or \"three\" or {\"/state/scalar/value/number/complex\": [3.14, -1.414]"
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

            let mut map = Map::new();

            let id = Id::from_str(&key).map_err(de::Error::custom)?;
            let txn = self.txn.subcontext(id.clone());
            let value = access.next_value(txn).await?;
            map.insert(id, value);

            while let Some(id) = access.next_key::<Id>(()).await? {
                let txn = self.txn.subcontext(id.clone());
                let state = access.next_value(txn).await?;
                map.insert(id, state);
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
impl<Txn, FE> de::FromStream for State<Txn, FE>
where
    Txn: Transaction<FE> + RPCClient<State<Txn, FE>>,
    FE: AsType<BTreeNode>
        + AsType<TensorNode>
        + AsType<ChainBlock>
        + DenseCacheFile
        + for<'b> FileSave<'b>
        + Clone,
{
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        let scalar = ScalarVisitor::default();
        decoder
            .decode_any(StateVisitor {
                txn,
                scalar,
                phantom: PhantomData,
            })
            .await
    }
}
