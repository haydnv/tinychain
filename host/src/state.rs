use std::collections::{HashMap, HashSet};
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::iter::FromIterator;
use std::str::FromStr;

use async_trait::async_trait;
use destream::{de, en};
use futures::future::try_join_all;
use futures::{stream, StreamExt, TryFutureExt};
use log::debug;
use safecast::TryCastFrom;
use transact::{IntoView, Transaction};

use error::*;
use generic::*;

use crate::chain::*;
use crate::fs::Dir;
use crate::object::{Object, ObjectType};
use crate::scalar::*;
use crate::txn::Txn;

#[derive(Clone, Eq, PartialEq)]
pub enum StateType {
    Chain(ChainType),
    Map,
    Object(ObjectType),
    Scalar(ScalarType),
    Tuple,
}

impl Class for StateType {
    type Instance = State;
}

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
            Self::Chain(ct) => ct.path(),
            Self::Map => path_label(&["state", "map"]).into(),
            Self::Object(ot) => ot.path(),
            Self::Scalar(st) => st.path(),
            Self::Tuple => path_label(&["state", "tuple"]).into(),
        }
    }
}

impl From<ChainType> for StateType {
    fn from(ct: ChainType) -> Self {
        Self::Chain(ct)
    }
}

impl From<ValueType> for StateType {
    fn from(vt: ValueType) -> Self {
        Self::Scalar(vt.into())
    }
}

impl fmt::Display for StateType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Chain(ct) => fmt::Display::fmt(ct, f),
            Self::Map => f.write_str("Map<Id, State>"),
            Self::Object(ot) => fmt::Display::fmt(ot, f),
            Self::Scalar(st) => fmt::Display::fmt(st, f),
            Self::Tuple => f.write_str("Tuple<State>"),
        }
    }
}

#[derive(Clone)]
pub enum State {
    Chain(Chain),
    Map(Map<Self>),
    Object(Object),
    Scalar(Scalar),
    Tuple(Tuple<Self>),
}

impl State {
    pub fn is_none(&self) -> bool {
        match self {
            Self::Scalar(scalar) => scalar.is_none(),
            Self::Tuple(tuple) => tuple.is_empty(),
            _ => false,
        }
    }

    pub fn is_ref(&self) -> bool {
        match self {
            Self::Map(map) => map.values().any(Self::is_ref),
            Self::Scalar(scalar) => scalar.is_ref(),
            Self::Tuple(tuple) => tuple.iter().any(Self::is_ref),
            _ => false,
        }
    }

    pub fn into_type(self, class: StateType) -> Option<Self> {
        if self.class() == class {
            return Some(self);
        }

        match class {
            StateType::Scalar(class) => {
                debug!("cast into {} from {}", class, self);
                Scalar::opt_cast_from(self)
                    .and_then(|scalar| scalar.into_type(class))
                    .map(Self::Scalar)
            }
            _ => None,
        }
    }
}

#[async_trait]
impl Refer for State {
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

    async fn resolve(self, context: &Map<State>, txn: &Txn) -> TCResult<Self> {
        match self {
            Self::Map(map) => {
                let resolved = try_join_all(
                    map.into_iter()
                        .map(|(id, state)| state.resolve(context, txn).map_ok(|s| (id, s))),
                )
                .await?;

                let map = HashMap::from_iter(resolved);
                Ok(State::Map(map.into()))
            }
            Self::Scalar(scalar) => scalar.resolve(context, txn).await,
            Self::Tuple(tuple) => {
                let resolved =
                    try_join_all(tuple.into_iter().map(|state| state.resolve(context, txn)))
                        .await?;

                Ok(State::Tuple(resolved.into()))
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
            Self::Map(_) => StateType::Map,
            Self::Object(object) => StateType::Object(object.class()),
            Self::Scalar(scalar) => StateType::Scalar(scalar.class()),
            Self::Tuple(_) => StateType::Tuple,
        }
    }
}

impl<'en> IntoView<'en, Dir> for State {
    type Txn = Txn;
    type View = StateView;

    fn into_view(self, txn: Txn) -> StateView {
        StateView { state: self, txn }
    }
}

impl From<()> for State {
    fn from(_: ()) -> State {
        State::Scalar(Scalar::Value(Value::None))
    }
}

impl From<Map<State>> for State {
    fn from(map: Map<State>) -> State {
        State::Map(map)
    }
}

impl From<OpRef> for State {
    fn from(op_ref: OpRef) -> Self {
        TCRef::Op(op_ref).into()
    }
}

impl From<Scalar> for State {
    fn from(scalar: Scalar) -> State {
        State::Scalar(scalar)
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
    fn from(value: Value) -> State {
        State::Scalar(value.into())
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

impl<T: Clone + TryCastFrom<State>> TryCastFrom<State> for Map<T> {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Map(map) => HashMap::<Id, T>::can_cast_from(map),
            State::Tuple(tuple) => Map::<T>::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Map(map) => HashMap::<Id, T>::opt_cast_from(map).map(Map::from),
            State::Tuple(tuple) => Map::<T>::opt_cast_from(tuple),
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

impl<T: Clone + TryCastFrom<State>> TryCastFrom<State> for Tuple<T> {
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

impl TryCastFrom<State> for Scalar {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Map(map) => HashMap::<Id, Scalar>::can_cast_from(map),
            State::Scalar(_) => true,
            State::Tuple(tuple) => Vec::<Scalar>::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Map(map) => HashMap::<Id, Scalar>::opt_cast_from(map)
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

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Chain(chain) => fmt::Display::fmt(chain, f),
            Self::Map(map) => fmt::Display::fmt(map, f),
            Self::Object(object) => fmt::Display::fmt(object, f),
            Self::Scalar(scalar) => fmt::Display::fmt(scalar, f),
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
            StateType::Chain(_ct) => {
                Err(de::Error::custom("decoding a Chain is not yet implemented"))
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
            log::debug!("deserialize: key is {}", key);

            if let Ok(path) = TCPathBuf::from_str(&key) {
                if let Some(class) = StateType::from_path(&path) {
                    if let Ok(state) = self.visit_map_value(class, &mut access).await {
                        return Ok(state);
                    }
                }
            }

            if let Ok(subject) = reference::Subject::from_str(&key) {
                let params = access.next_value(()).await?;
                return ScalarVisitor::visit_subject(subject, params).map(State::Scalar);
            }

            let mut map = HashMap::new();

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

pub struct StateView {
    state: State,
    txn: Txn,
}

impl<'en> en::IntoStream<'en> for StateView {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self.state {
            State::Chain(_chain) => {
                Err(en::Error::custom("encoding a Chain is not yet implemented"))
            }
            State::Map(map) => {
                let txn = self.txn.clone();
                let map = stream::iter(map.into_iter())
                    .map(move |(id, state)| (id, state.into_view(txn.clone())))
                    .map(Ok);

                encoder.encode_map_stream(map)
            }
            State::Object(object) => object.into_view(self.txn).into_stream(encoder),
            State::Scalar(scalar) => scalar.into_stream(encoder),
            State::Tuple(tuple) => {
                let txn = self.txn.clone();
                let map = stream::iter(tuple.into_iter())
                    .map(move |state| state.into_view(txn.clone()))
                    .map(Ok);

                encoder.encode_seq_stream(map)
            }
        }
    }
}
