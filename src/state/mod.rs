use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use async_trait::async_trait;
use destream::de::{self, Decoder, FromStream, MapAccess, SeqAccess, Visitor};
use destream::en::{Encoder, IntoStream, ToStream};
use log::debug;
use safecast::{Match, TryCastFrom, TryCastInto};

use error::*;
use generic::*;

use crate::scalar::*;

pub mod reference;

pub use reference::*;

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum StateType {
    Map,
    Ref(RefType),
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
            } else if path.len() > 2 && &path[1] == "scalar" {
                ScalarType::from_path(path).map(Self::Scalar)
            } else if path.len() > 2 && &path[1] == "ref" {
                RefType::from_path(path).map(Self::Ref)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        match self {
            Self::Map => path_label(&["state", "map"]).into(),
            Self::Ref(rt) => rt.path(),
            Self::Scalar(st) => st.path(),
            Self::Tuple => path_label(&["state", "tuple"]).into(),
        }
    }
}

impl fmt::Display for StateType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Map => f.write_str("Map<Id, State>"),
            Self::Ref(rt) => fmt::Display::fmt(rt, f),
            Self::Scalar(st) => fmt::Display::fmt(st, f),
            Self::Tuple => f.write_str("Tuple<State>"),
        }
    }
}

#[derive(Clone)]
pub enum State {
    Map(Map<Self>),
    Ref(Box<TCRef>),
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

    pub fn into_type(self, class: StateType) -> TCResult<Self> {
        use OpDefType as ODT;
        use OpRefType as ORT;
        use RefType as RT;
        use ScalarType as ST;
        use ValueType as VT;

        match class {
            StateType::Scalar(class) => match class {
                ST::Map => Err(cast_err(ST::Map, &self)),
                ST::Op(ot) => match ot {
                    ODT::Get => {
                        let op = self.try_cast_into(try_cast_err(ODT::Get))?;
                        Ok(Scalar::Op(OpDef::Get(op)).into())
                    }
                    ODT::Put => {
                        let op = self.try_cast_into(try_cast_err(ODT::Put))?;
                        Ok(Scalar::Op(OpDef::Put(op)).into())
                    }
                    ODT::Post => {
                        let op = self.try_cast_into(try_cast_err(ODT::Post))?;
                        Ok(Scalar::Op(OpDef::Post(op)).into())
                    }
                    ODT::Delete => {
                        let op = self.try_cast_into(try_cast_err(ODT::Delete))?;
                        Ok(Scalar::Op(OpDef::Delete(op)).into())
                    }
                },
                ScalarType::Value(class) => match class {
                    VT::Link => {
                        let l = Link::try_cast_from(self, try_cast_err(VT::Link))?;
                        Ok(Value::Link(l).into())
                    }
                    VT::None => Ok(Value::None.into()),
                    VT::Number(nt) => {
                        let v: Value = self.try_cast_into(try_cast_err(nt))?;
                        let n: Number = v.try_cast_into(try_cast_err(nt))?;
                        let n = n.into_type(nt);
                        Ok(Value::Number(n).into())
                    }
                    VT::String => Ok(Value::String(self.to_string()).into()),
                    VT::Tuple => {
                        let tuple = Tuple::<Value>::try_cast_from(self, try_cast_err(ST::Tuple))?;
                        Ok(Value::Tuple(tuple).into())
                    }
                    VT::Value => Ok(self.into()),
                },
                ST::Tuple => {
                    let tuple = Tuple::<Scalar>::try_cast_from(self, try_cast_err(ST::Tuple))?;
                    Ok(Scalar::Tuple(tuple).into())
                }
            },
            StateType::Ref(rt) => match rt {
                RT::Id => {
                    let id_ref = IdRef::try_cast_from(self, try_cast_err(RT::Id))?;
                    Ok(State::from(id_ref))
                }
                RT::Op(ort) => match ort {
                    ORT::Get => {
                        let get_ref = GetRef::try_cast_from(self, try_cast_err(ORT::Get))?;
                        Ok(State::from(OpRef::Get(get_ref)))
                    }
                    ORT::Put => {
                        let put_ref = PutRef::try_cast_from(self, try_cast_err(ORT::Put))?;
                        Ok(State::from(OpRef::Put(put_ref)))
                    }
                    ORT::Post => {
                        let post_ref = PostRef::try_cast_from(self, try_cast_err(ORT::Put))?;
                        Ok(State::from(OpRef::Post(post_ref)))
                    }
                    ORT::Delete => {
                        let delete_ref = DeleteRef::try_cast_from(self, try_cast_err(ORT::Put))?;
                        Ok(State::from(OpRef::Delete(delete_ref)))
                    }
                },
            },
            other => Err(TCError::bad_request(
                format!("Cannot cast into {} from", other),
                self,
            )),
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
            Self::Map(_) => StateType::Map,
            Self::Ref(tc_ref) => StateType::Ref(tc_ref.class()),
            Self::Scalar(scalar) => StateType::Scalar(scalar.class()),
            Self::Tuple(_) => StateType::Tuple,
        }
    }
}

impl From<IdRef> for State {
    fn from(id_ref: IdRef) -> Self {
        TCRef::Id(id_ref).into()
    }
}

impl From<OpRef> for State {
    fn from(op_ref: OpRef) -> Self {
        TCRef::Op(op_ref).into()
    }
}

impl From<TCRef> for State {
    fn from(tc_ref: TCRef) -> Self {
        Self::Ref(Box::new(tc_ref))
    }
}

impl From<Scalar> for State {
    fn from(scalar: Scalar) -> State {
        State::Scalar(scalar)
    }
}

impl From<Value> for State {
    fn from(value: Value) -> State {
        State::Scalar(value.into())
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
            State::Ref(tc_ref) => match &**tc_ref {
                TCRef::Id(_) => true,
                _ => false,
            },
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Ref(tc_ref) => match *tc_ref {
                TCRef::Id(id_ref) => Some(id_ref),
                _ => None,
            },
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
            State::Scalar(_) => None,
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
            State::Map(_) => false,
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            State::Tuple(tuple) => tuple.iter().all(Self::can_cast_from),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Map(_) => None,
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
            Self::Map(map) => fmt::Display::fmt(map, f),
            Self::Ref(tc_ref) => fmt::Display::fmt(tc_ref, f),
            Self::Scalar(scalar) => fmt::Display::fmt(scalar, f),
            Self::Tuple(tuple) => fmt::Display::fmt(tuple, f),
        }
    }
}

#[derive(Default)]
struct StateVisitor {
    scalar: ScalarVisitor,
}

#[async_trait]
impl Visitor for StateVisitor {
    type Value = State;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a Tinychain State, e.g. 1 or [2] or \"three\" or {\"/state/scalar/value/number/complex\": [3.14, -1.414]")
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

    async fn visit_map<A: MapAccess>(self, mut access: A) -> Result<Self::Value, A::Error> {
        if let Some(key) = access.next_key::<String>().await? {
            log::debug!("deserialize: key is {}", key);

            if let Ok(subject) = Subject::from_str(&key) {
                if let Subject::Link(link) = &subject {
                    if link.host().is_none() {
                        if let Some(class) = StateType::from_path(link.path()) {
                            let value: State = access.next_value().await?;
                            return if value.matches::<(IdRef,)>() {
                                let (value,) = value.opt_cast_into().unwrap();
                                Ok(OpRef::Get((
                                    Subject::Link(Link::from(class.path())),
                                    Key::Ref(value),
                                ))
                                .into())
                            } else {
                                value.into_type(class).map_err(de::Error::custom)
                            };
                        }
                    }
                }

                let params: State = access.next_value().await?;
                return if params.is_none() {
                    match subject {
                        Subject::Link(link) => Ok(Value::Link(link).into()),
                        Subject::Ref(id_ref) => Ok(State::from(id_ref)),
                    }
                } else {
                    RefVisitor::visit_ref_value(subject, params).map(State::from)
                };
            }

            let key = Id::from_str(&key).map_err(de::Error::custom)?;

            let mut map = HashMap::new();
            let value = access.next_value().await?;
            map.insert(key, value);

            while let Some(key) = access.next_key().await? {
                let value = access.next_value().await?;
                map.insert(key, value);
            }

            Ok(State::Map(map.into()))
        } else {
            Ok(State::Map(Map::default()))
        }
    }

    async fn visit_seq<A: SeqAccess>(self, mut access: A) -> Result<Self::Value, A::Error> {
        let mut seq = if let Some(len) = access.size_hint() {
            Vec::with_capacity(len)
        } else {
            Vec::new()
        };

        while let Some(next) = access.next_element().await? {
            seq.push(next);
        }

        Ok(State::Tuple(seq.into()))
    }
}

#[async_trait]
impl FromStream for State {
    async fn from_stream<D: Decoder>(decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_any(StateVisitor::default()).await
    }
}

impl<'en> ToStream<'en> for State {
    fn to_stream<E: Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Map(map) => map.to_stream(encoder),
            Self::Ref(tc_ref) => tc_ref.to_stream(encoder),
            Self::Scalar(scalar) => scalar.to_stream(encoder),
            Self::Tuple(tuple) => tuple.to_stream(encoder),
        }
    }
}

impl<'en> IntoStream<'en> for State {
    fn into_stream<E: Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Map(map) => map.into_inner().into_stream(encoder),
            Self::Ref(tc_ref) => tc_ref.into_stream(encoder),
            Self::Scalar(scalar) => scalar.into_stream(encoder),
            Self::Tuple(tuple) => tuple.into_inner().into_stream(encoder),
        }
    }
}

fn cast_err<F: fmt::Display, T: fmt::Display>(to: T, from: &F) -> TCError {
    TCError::bad_request(format!("cannot cast into {} from", to), from)
}

fn try_cast_err<F: fmt::Display, T: fmt::Display>(to: T) -> impl FnOnce(&F) -> TCError {
    move |s| cast_err(to, s)
}
