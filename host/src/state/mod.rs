//! A Tinychain [`State`]

use std::collections::{BTreeMap, HashSet};
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::str::FromStr;

use async_trait::async_trait;
use bytes::Bytes;
use destream::de;
use futures::future::TryFutureExt;
use futures::stream::{StreamExt, TryStreamExt};
use log::debug;
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::Transaction;
use tcgeneric::*;

use crate::chain::*;
use crate::closure::*;
use crate::collection::*;
use crate::object::{Object, ObjectType};
use crate::route::Public;
use crate::scalar::*;
use crate::stream::TCStream;
use crate::txn::Txn;

pub use view::StateView;

mod view;

pub trait StateClass: Class
where
    State: From<Self::Get>,
{
    type Get;

    fn try_cast_from_value(self, value: Value) -> TCResult<Self::Get>;
}

/// The [`Class`] of a [`State`].
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum StateType {
    Chain(ChainType),
    Collection(CollectionType),
    Closure,
    Map,
    Object(ObjectType),
    Scalar(ScalarType),
    Stream,
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
                    "map" => Some(Self::Map),
                    "tuple" => Some(Self::Tuple),
                    _ => None,
                }
            } else if path.len() > 2 {
                match path[1].as_str() {
                    "collection" => CollectionType::from_path(path).map(Self::Collection),
                    "chain" => ChainType::from_path(path).map(Self::Chain),
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
            Self::Stream => path_label(&["state", "stream"]).into(),
            Self::Tuple => path_label(&["state", "tuple"]).into(),
        }
    }
}

impl StateClass for StateType {
    type Get = State;

    fn try_cast_from_value(self, value: Value) -> TCResult<Self::Get> {
        match self {
            Self::Chain(ct) => ct.try_cast_from_value(value).map(State::from),
            Self::Collection(ct) => ct.try_cast_from_value(value).map(State::from),
            Self::Closure => Err(TCError::bad_request("cannot cast into closure from", value)),
            Self::Map => {
                let contents: Tuple<Value> = value.try_into()?;
                let contents: Vec<(Id, Value)> = contents
                    .try_cast_into(|v| TCError::bad_request("cannot cast into Map from", v))?;

                Ok(State::Map(
                    contents
                        .into_iter()
                        .map(|(id, value)| (id, State::from(value)))
                        .collect(),
                ))
            }
            Self::Object(ot) => ot.try_cast_from_value(value).map(State::Object),
            Self::Scalar(st) => st.try_cast_from_value(value).map(State::Scalar),
            Self::Stream => Err(TCError::bad_request("cannot cast into Stream from", value)),
            Self::Tuple => {
                let contents: Tuple<Value> = value.try_into()?;
                Ok(State::Tuple(contents.into_iter().collect()))
            }
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

impl From<ChainType> for StateType {
    fn from(ct: ChainType) -> Self {
        Self::Chain(ct)
    }
}

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
            other => Err(TCError::bad_request(
                "expected Scalar class but found",
                other,
            )),
        }
    }
}

impl fmt::Display for StateType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Chain(ct) => fmt::Display::fmt(ct, f),
            Self::Collection(ct) => fmt::Display::fmt(ct, f),
            Self::Closure => f.write_str("closure"),
            Self::Map => f.write_str("Map<Id, State>"),
            Self::Object(ot) => fmt::Display::fmt(ot, f),
            Self::Scalar(st) => fmt::Display::fmt(st, f),
            Self::Stream => f.write_str("Stream"),
            Self::Tuple => f.write_str("Tuple<State>"),
        }
    }
}

/// An addressable state with a discrete value per-transaction.
#[derive(Clone)]
pub enum State {
    Collection(Collection),
    Chain(Chain),
    Closure(Closure),
    Map(Map<Self>),
    Object(Object),
    Scalar(Scalar),
    Stream(TCStream),
    Tuple(Tuple<Self>),
}

impl State {
    /// Return true if this `State` is an empty [`Tuple`] or [`Map`], default [`Link`], or `Value::None`
    pub fn is_none(&self) -> bool {
        match self {
            Self::Map(map) => map.is_empty(),
            Self::Scalar(scalar) => scalar.is_none(),
            Self::Tuple(tuple) => tuple.is_empty(),
            _ => false,
        }
    }

    /// Return true if this `State` variant is a [`Map`].
    pub fn is_map(&self) -> bool {
        match self {
            Self::Scalar(scalar) => scalar.is_map(),
            Self::Map(_) => true,
            _ => false,
        }
    }

    /// Return false if this `State` is an empty [`Tuple`] or [`Map`], default [`Link`], or `Value::None`
    pub fn is_some(&self) -> bool {
        !self.is_none()
    }

    /// Return true if this `State` variant is a [`Tuple`].
    pub fn is_tuple(&self) -> bool {
        match self {
            Self::Scalar(scalar) => scalar.is_tuple(),
            Self::Tuple(_) => true,
            _ => false,
        }
    }

    /// Return true if this `State` is a reference that needs to be resolved.
    pub fn is_ref(&self) -> bool {
        match self {
            Self::Map(map) => map.values().any(Self::is_ref),
            Self::Scalar(scalar) => scalar.is_ref(),
            Self::Tuple(tuple) => tuple.iter().any(Self::is_ref),
            _ => false,
        }
    }

    /// Return this `State` as a [`Map`] of [`State`]s, or an error if this is not possible.
    pub fn try_into_map<Err: Fn(State) -> TCError>(self, err: Err) -> TCResult<Map<State>> {
        match self {
            State::Map(states) => Ok(states),
            State::Scalar(Scalar::Map(states)) => Ok(states
                .into_iter()
                .map(|(id, scalar)| (id, State::Scalar(scalar)))
                .collect()),
            other => Err((err)(other)),
        }
    }

    pub fn try_into_tuple<Err: Fn(State) -> TCError>(self, err: Err) -> TCResult<Tuple<State>> {
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

#[async_trait]
impl Refer for State {
    fn dereference_self(self, path: &TCPathBuf) -> Self {
        match self {
            Self::Map(map) => {
                let map = map
                    .into_iter()
                    .map(|(id, state)| (id, state.dereference_self(path)))
                    .collect();

                Self::Map(map)
            }
            Self::Scalar(scalar) => Self::Scalar(scalar.dereference_self(path)),
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

    fn is_inter_service_write(&self, cluster_path: &[PathSegment]) -> bool {
        match self {
            Self::Map(map) => map
                .values()
                .any(|state| state.is_inter_service_write(cluster_path)),

            Self::Scalar(scalar) => scalar.is_inter_service_write(cluster_path),

            Self::Tuple(tuple) => tuple
                .iter()
                .any(|state| state.is_inter_service_write(cluster_path)),

            _ => false,
        }
    }

    fn reference_self(self, path: &TCPathBuf) -> Self {
        match self {
            Self::Map(map) => {
                let map = map
                    .into_iter()
                    .map(|(id, state)| (id, state.reference_self(path)))
                    .collect();

                Self::Map(map)
            }
            Self::Scalar(scalar) => Self::Scalar(scalar.reference_self(path)),
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
            Self::Scalar(scalar) => scalar.requires(deps),
            Self::Tuple(tuple) => {
                for state in tuple.iter() {
                    state.requires(deps);
                }
            }
            _ => {}
        }
    }

    async fn resolve<'a, T: Instance + Public>(
        self,
        context: &'a Scope<'a, T>,
        txn: &'a Txn,
    ) -> TCResult<Self> {
        debug!("State::resolve {}", self);

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

impl Default for State {
    fn default() -> Self {
        Self::Scalar(Scalar::default())
    }
}

impl Instance for State {
    type Class = StateType;

    fn class(&self) -> StateType {
        match self {
            Self::Chain(chain) => StateType::Chain(chain.class()),
            Self::Closure(_) => StateType::Closure,
            Self::Collection(collection) => StateType::Collection(collection.class()),
            Self::Map(_) => StateType::Map,
            Self::Object(object) => StateType::Object(object.class()),
            Self::Scalar(scalar) => StateType::Scalar(scalar.class()),
            Self::Stream(_) => StateType::Stream,
            Self::Tuple(_) => StateType::Tuple,
        }
    }
}

impl From<()> for State {
    fn from(_: ()) -> State {
        State::Scalar(Scalar::Value(Value::None))
    }
}

impl From<BTree> for State {
    fn from(btree: BTree) -> Self {
        Self::Collection(btree.into())
    }
}

impl From<Chain> for State {
    fn from(chain: Chain) -> Self {
        Self::Chain(chain)
    }
}

impl From<Closure> for State {
    fn from(closure: Closure) -> Self {
        Self::Closure(closure)
    }
}

impl From<Collection> for State {
    fn from(collection: Collection) -> Self {
        Self::Collection(collection)
    }
}

impl From<Link> for State {
    fn from(link: Link) -> Self {
        Self::Scalar(Scalar::from(link))
    }
}

impl From<Map<State>> for State {
    fn from(map: Map<State>) -> Self {
        State::Map(map)
    }
}

impl From<Map<Scalar>> for State {
    fn from(map: Map<Scalar>) -> Self {
        State::Scalar(map.into())
    }
}

impl From<Object> for State {
    fn from(object: Object) -> Self {
        State::Object(object)
    }
}

impl From<OpDef> for State {
    fn from(op_def: OpDef) -> Self {
        State::Scalar(Scalar::Op(op_def))
    }
}

impl From<OpRef> for State {
    fn from(op_ref: OpRef) -> Self {
        TCRef::Op(op_ref).into()
    }
}

impl From<Scalar> for State {
    fn from(scalar: Scalar) -> Self {
        State::Scalar(scalar)
    }
}

impl From<Table> for State {
    fn from(table: Table) -> Self {
        Self::Collection(table.into())
    }
}

impl From<TCStream> for State {
    fn from(stream: TCStream) -> Self {
        Self::Stream(stream)
    }
}

#[cfg(feature = "tensor")]
impl From<Tensor> for State {
    fn from(tensor: Tensor) -> Self {
        Self::Collection(tensor.into())
    }
}

impl From<Tuple<State>> for State {
    fn from(tuple: Tuple<State>) -> Self {
        Self::Tuple(tuple)
    }
}

impl From<Tuple<Scalar>> for State {
    fn from(tuple: Tuple<Scalar>) -> Self {
        Self::Scalar(tuple.into())
    }
}

impl From<Tuple<Value>> for State {
    fn from(tuple: Tuple<Value>) -> Self {
        Self::Scalar(tuple.into())
    }
}

impl From<TCRef> for State {
    fn from(tc_ref: TCRef) -> Self {
        Box::new(tc_ref).into()
    }
}

impl From<Box<TCRef>> for State {
    fn from(tc_ref: Box<TCRef>) -> Self {
        Self::Scalar(Scalar::Ref(tc_ref))
    }
}

impl From<Value> for State {
    fn from(value: Value) -> Self {
        Self::Scalar(value.into())
    }
}

impl From<bool> for State {
    fn from(b: bool) -> Self {
        Self::Scalar(b.into())
    }
}

impl From<u64> for State {
    fn from(n: u64) -> Self {
        Self::Scalar(n.into())
    }
}

impl TryFrom<State> for Map<State> {
    type Error = TCError;

    fn try_from(state: State) -> TCResult<Map<State>> {
        match state {
            State::Map(map) => Ok(map),
            other => Err(TCError::bad_request("expected Map but found", other)),
        }
    }
}

impl TryFrom<State> for Value {
    type Error = TCError;

    fn try_from(state: State) -> TCResult<Value> {
        match state {
            State::Scalar(scalar) => scalar.try_into(),
            other => Err(TCError::bad_request("expected Value but found", other)),
        }
    }
}

impl TryCastFrom<State> for bool {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Scalar(scalar) => Self::opt_cast_from(scalar),
            _ => None,
        }
    }
}

impl TryCastFrom<State> for Bytes {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Scalar(scalar) => Self::opt_cast_from(scalar),
            _ => None,
        }
    }
}

impl TryCastFrom<State> for Closure {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Closure(_) => true,
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Closure(closure) => Some(closure),
            State::Scalar(scalar) => Self::opt_cast_from(scalar),
            _ => None,
        }
    }
}

impl TryCastFrom<State> for String {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Scalar(scalar) => Self::opt_cast_from(scalar),
            _ => None,
        }
    }
}

impl<T: Clone + TryCastFrom<State>> TryCastFrom<State> for Map<T> {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Map(map) => BTreeMap::<Id, T>::can_cast_from(map),
            State::Tuple(tuple) => Map::<T>::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Map(map) => BTreeMap::<Id, T>::opt_cast_from(map).map(Map::from),
            State::Tuple(tuple) => Map::<T>::opt_cast_from(tuple),
            _ => None,
        }
    }
}

impl TryCastFrom<State> for TCStream {
    fn can_cast_from(state: &State) -> bool {
        match state {
            // State::Chain(_) => true, // TODO
            State::Collection(_) => true,
            State::Stream(_) => true,
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Collection(collection) => Some(Self::from(collection)),
            State::Stream(stream) => Some(stream),
            _ => None,
        }
    }
}

impl<T: TryCastFrom<State>> TryCastFrom<State> for (T,) {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Tuple(tuple) => Self::opt_cast_from(tuple),
            _ => None,
        }
    }
}

impl<T1: TryCastFrom<State>, T2: TryCastFrom<State>> TryCastFrom<State> for (T1, T2) {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Tuple(tuple) => Self::opt_cast_from(tuple),
            _ => None,
        }
    }
}

impl<T1: TryCastFrom<State>, T2: TryCastFrom<State>, T3: TryCastFrom<State>> TryCastFrom<State>
    for (T1, T2, T3)
{
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Tuple(tuple) => Self::opt_cast_from(tuple),
            _ => None,
        }
    }
}

impl<T: TryCastFrom<State>> TryCastFrom<State> for Vec<T> {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Tuple(source) => Self::opt_cast_from(source),
            _ => None,
        }
    }
}

impl<T: TryCastFrom<State>> TryCastFrom<State> for Tuple<T> {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Tuple(tuple) => Vec::<T>::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Tuple(tuple) => Vec::<T>::opt_cast_from(tuple).map(Tuple::from),
            _ => None,
        }
    }
}

impl TryCastFrom<State> for Id {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Scalar(scalar) => Self::opt_cast_from(scalar),
            _ => None,
        }
    }
}

impl TryCastFrom<State> for IdRef {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Scalar(scalar) => Self::opt_cast_from(scalar),
            _ => None,
        }
    }
}

impl TryCastFrom<State> for Link {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Scalar(scalar) => Self::opt_cast_from(scalar),
            _ => None,
        }
    }
}

impl TryCastFrom<State> for OpDef {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Scalar(scalar) => Self::opt_cast_from(scalar),
            _ => None,
        }
    }
}

impl TryCastFrom<State> for OpRef {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Scalar(scalar) => Self::opt_cast_from(scalar),
            _ => None,
        }
    }
}

impl TryCastFrom<State> for Scalar {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Map(map) => BTreeMap::<Id, Scalar>::can_cast_from(map),
            State::Scalar(_) => true,
            State::Tuple(tuple) => Vec::<Scalar>::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Map(map) => BTreeMap::<Id, Scalar>::opt_cast_from(map)
                .map(Map::from)
                .map(Scalar::Map),

            State::Scalar(scalar) => Some(scalar),

            State::Tuple(tuple) => Vec::<Scalar>::opt_cast_from(tuple)
                .map(Tuple::from)
                .map(Scalar::Tuple),

            _ => None,
        }
    }
}

#[cfg(feature = "tensor")]
impl TryCastFrom<State> for Tensor {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Collection(collection) => Self::can_cast_from(collection),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Collection(collection) => Self::opt_cast_from(collection),
            _ => None,
        }
    }
}

impl TryCastFrom<State> for TCPathBuf {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Scalar(scalar) => Self::opt_cast_from(scalar),
            _ => None,
        }
    }
}

impl TryCastFrom<State> for Value {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            State::Tuple(tuple) => tuple.iter().all(Self::can_cast_from),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Scalar(scalar) => Self::opt_cast_from(scalar),

            State::Tuple(tuple) => Vec::<Value>::opt_cast_from(tuple)
                .map(Tuple::from)
                .map(Value::Tuple),

            _ => None,
        }
    }
}

impl TryCastFrom<State> for Number {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Scalar(scalar) => Self::opt_cast_from(scalar),
            _ => None,
        }
    }
}

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Chain(chain) => fmt::Debug::fmt(chain, f),
            Self::Closure(closure) => fmt::Debug::fmt(closure, f),
            Self::Collection(collection) => fmt::Debug::fmt(collection, f),
            Self::Map(map) => fmt::Debug::fmt(map, f),
            Self::Object(object) => fmt::Debug::fmt(object, f),
            Self::Scalar(scalar) => fmt::Debug::fmt(scalar, f),
            Self::Stream(_) => f.write_str("Stream"),
            Self::Tuple(tuple) => fmt::Debug::fmt(tuple, f),
        }
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Chain(chain) => fmt::Display::fmt(chain, f),
            Self::Closure(closure) => fmt::Display::fmt(closure, f),
            Self::Collection(collection) => fmt::Display::fmt(collection, f),
            Self::Map(map) => fmt::Display::fmt(map, f),
            Self::Object(object) => fmt::Display::fmt(object, f),
            Self::Scalar(scalar) => fmt::Display::fmt(scalar, f),
            Self::Stream(_) => f.write_str("Stream"),
            Self::Tuple(tuple) => fmt::Display::fmt(tuple, f),
        }
    }
}

struct StateVisitor {
    txn: Txn,
    scalar: ScalarVisitor,
}

impl StateVisitor {
    async fn visit_map_value<A: de::MapAccess>(
        &self,
        class: StateType,
        access: &mut A,
    ) -> Result<State, A::Error> {
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
                    .map_ok(State::Collection)
                    .await
            }
            StateType::Map => access.next_value(self.txn.clone()).await,
            StateType::Object(ot) => match ot {
                ObjectType::Class => {
                    access
                        .next_value(())
                        .map_ok(Object::Class)
                        .map_ok(State::Object)
                        .await
                }
                ObjectType::Instance => {
                    let op_ref = access.next_value(()).map_ok(TCRef::Op).await?;
                    Ok(State::Scalar(Scalar::Ref(op_ref.into())))
                }
            },
            StateType::Scalar(st) => {
                ScalarVisitor::visit_map_value(st, access)
                    .map_ok(State::Scalar)
                    .await
            }
            StateType::Stream => Err(de::Error::invalid_type("Stream", "a Collection")),
            StateType::Tuple => access.next_value(self.txn.clone()).await,
        }
    }
}

#[async_trait]
impl<'a> de::Visitor for StateVisitor {
    type Value = State;

    fn expecting() -> &'static str {
        "a State, e.g. 1 or [2] or \"three\" or {\"/state/scalar/value/number/complex\": [3.14, -1.414]"
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

    fn visit_byte_buf<E: de::Error>(self, buf: Vec<u8>) -> Result<Self::Value, E> {
        self.scalar.visit_byte_buf(buf).map(State::Scalar)
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
                        debug!("deserialize instance of {}...", class);
                        return self.visit_map_value(class, &mut access).await;
                    } else {
                        debug!("not a classpath: {}", path);
                    }
                }
            }

            debug!("deserialize Op with subject {}", key);
            if let Ok(subject) = reference::Subject::from_str(&key) {
                let params = access.next_value(()).await?;
                return ScalarVisitor::visit_subject(subject, params).map(State::Scalar);
            }

            let mut map = BTreeMap::new();

            let id = Id::from_str(&key).map_err(de::Error::custom)?;
            let txn = self
                .txn
                .subcontext(id.clone())
                .map_err(de::Error::custom)
                .await?;

            let value = access.next_value(txn).await?;
            map.insert(id, value);

            while let Some(id) = access.next_key::<Id>(()).await? {
                let txn = self
                    .txn
                    .subcontext(id.clone())
                    .map_err(de::Error::custom)
                    .await?;

                let state = access.next_value(txn).await?;
                map.insert(id, state);
            }

            Ok(State::Map(map.into()))
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
            let txn = self
                .txn
                .subcontext(i.into())
                .map_err(de::Error::custom)
                .await?;

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
impl de::FromStream for State {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        let scalar = ScalarVisitor::default();
        decoder.decode_any(StateVisitor { txn, scalar }).await
    }
}
