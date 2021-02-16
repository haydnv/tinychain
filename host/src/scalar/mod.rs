//! Immutable values which always reside in memory

use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::fmt;
use std::iter::FromIterator;
use std::str::FromStr;

use async_trait::async_trait;
use destream::de::{self, Decoder, FromStream};
use destream::en::{Encoder, IntoStream, ToStream};
use futures::future::{try_join_all, TryFutureExt};
use log::debug;
use safecast::{Match, TryCastFrom, TryCastInto};

use error::*;
use generic::*;

use crate::state::State;
use crate::txn::Txn;

pub mod op;
pub mod reference;

pub use op::*;
pub use reference::*;
pub use value::*;

const PREFIX: PathLabel = path_label(&["state", "scalar"]);

/// The [`Class`] of a [`Scalar`].
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum ScalarType {
    Map,
    Op(OpDefType),
    Ref(RefType),
    Tuple,
    Value(ValueType),
}

impl Class for ScalarType {
    type Instance = Scalar;
}

impl NativeClass for ScalarType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        debug!("ScalarType::from_path {}", TCPath::from(path));

        if path.len() > 2 && &path[..2] == &PREFIX[..] {
            match path[2].as_str() {
                "map" if path.len() == 3 => Some(Self::Map),
                "op" => OpDefType::from_path(path).map(Self::Op),
                "ref" => RefType::from_path(path).map(Self::Ref),
                "tuple" if path.len() == 3 => Some(Self::Tuple),
                "value" => ValueType::from_path(path).map(Self::Value),
                _ => None,
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        let prefix = TCPathBuf::from(PREFIX);

        match self {
            Self::Map => prefix.append(label("map")),
            Self::Op(odt) => odt.path(),
            Self::Ref(rt) => rt.path(),
            Self::Value(vt) => vt.path(),
            Self::Tuple => prefix.append(label("tuple")),
        }
    }
}

impl From<ValueType> for ScalarType {
    fn from(vt: ValueType) -> Self {
        Self::Value(vt)
    }
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Map => f.write_str("Map<Scalar>"),
            Self::Op(odt) => fmt::Display::fmt(odt, f),
            Self::Ref(rt) => fmt::Display::fmt(rt, f),
            Self::Value(vt) => fmt::Display::fmt(vt, f),
            Self::Tuple => f.write_str("Tuple<Scalar>"),
        }
    }
}

/// A scalar value, i.e. one which is always held in main memory and never split into blocks.
#[derive(Clone, Eq, PartialEq)]
pub enum Scalar {
    Map(Map<Self>),
    Op(OpDef),
    Ref(Box<TCRef>),
    Tuple(Tuple<Self>),
    Value(Value),
}

impl Scalar {
    /// Return true if self is an empty tuple, default link, or `Value::None`.
    pub fn is_none(&self) -> bool {
        match self {
            Self::Tuple(tuple) => tuple.is_empty(),
            Self::Value(value) => value.is_none(),
            _ => false,
        }
    }

    /// Return true if self is a reference type which needs to be resolved.
    pub fn is_ref(&self) -> bool {
        match self {
            Self::Map(map) => map.values().any(Self::is_ref),
            Self::Ref(_) => true,
            Self::Tuple(tuple) => tuple.iter().any(Self::is_ref),
            _ => false,
        }
    }

    /// Cast this `Scalar` into the specified [`ScalarType`], if possible.
    pub fn into_type(self, class: ScalarType) -> Option<Self> {
        if self.class() == class {
            return Some(self);
        }

        use OpDefType as ODT;
        use OpRefType as ORT;
        use RefType as RT;
        use ScalarType as ST;

        match class {
            ST::Map => self.opt_cast_into().map(Self::Map),
            ST::Op(odt) => match odt {
                ODT::Get => self.opt_cast_into().map(OpDef::Get).map(Self::Op),

                ODT::Put => self.opt_cast_into().map(OpDef::Put).map(Self::Op),

                ODT::Post => self.opt_cast_into().map(OpDef::Post).map(Self::Op),

                ODT::Delete => self.opt_cast_into().map(OpDef::Delete).map(Self::Op),
            },
            ST::Ref(rt) => match rt {
                RT::After => self
                    .opt_cast_into()
                    .map(Box::new)
                    .map(TCRef::After)
                    .map(Box::new)
                    .map(Scalar::Ref),

                RT::Id => self
                    .opt_cast_into()
                    .map(TCRef::Id)
                    .map(Box::new)
                    .map(Scalar::Ref),

                RT::If => self
                    .opt_cast_into()
                    .map(Box::new)
                    .map(TCRef::If)
                    .map(Box::new)
                    .map(Scalar::Ref),

                RT::Op(ort) => {
                    let op_ref = match ort {
                        ORT::Get => self.opt_cast_into().map(OpRef::Get),
                        ORT::Put => self.opt_cast_into().map(OpRef::Put),
                        ORT::Post => self.opt_cast_into().map(OpRef::Post),
                        ORT::Delete => self.opt_cast_into().map(OpRef::Delete),
                    };

                    op_ref.map(TCRef::Op).map(Self::from)
                }
            },
            ST::Value(vt) => Value::opt_cast_from(self)
                .and_then(|value| value.into_type(vt))
                .map(Scalar::Value),
            ST::Tuple => match self {
                Self::Map(map) => Some(Self::Tuple(map.into_iter().map(|(_, v)| v).collect())),
                Self::Tuple(tuple) => Some(Self::Tuple(tuple)),
                _ => None,
            },
        }
    }
}

impl Default for Scalar {
    fn default() -> Self {
        Self::Value(Value::default())
    }
}

impl Instance for Scalar {
    type Class = ScalarType;

    fn class(&self) -> ScalarType {
        use ScalarType as ST;
        match self {
            Self::Map(_) => ST::Map,
            Self::Op(op) => ST::Op(op.class()),
            Self::Ref(tc_ref) => ST::Ref(tc_ref.class()),
            Self::Tuple(_) => ST::Tuple,
            Self::Value(value) => ST::Value(value.class()),
        }
    }
}

#[async_trait]
impl Refer for Scalar {
    fn requires(&self, deps: &mut HashSet<Id>) {
        match self {
            Self::Map(map) => {
                for scalar in map.values() {
                    scalar.requires(deps);
                }
            }
            Self::Ref(tc_ref) => tc_ref.requires(deps),
            Self::Tuple(tuple) => {
                for scalar in tuple.iter() {
                    scalar.requires(deps);
                }
            }
            _ => {}
        }
    }

    async fn resolve(self, context: &Map<State>, txn: &Txn) -> TCResult<State> {
        match self {
            Self::Map(map) => {
                let resolved =
                    try_join_all(map.into_iter().map(|(id, scalar)| {
                        scalar.resolve(context, txn).map_ok(|state| (id, state))
                    }))
                    .await?;
                Ok(State::Map(Map::from_iter(resolved)))
            }
            Self::Ref(tc_ref) => tc_ref.resolve(context, txn).await,
            Self::Tuple(tuple) => {
                let resolved =
                    try_join_all(tuple.into_iter().map(|scalar| scalar.resolve(context, txn)))
                        .await?;
                Ok(State::Tuple(resolved.into()))
            }
            other => Ok(State::Scalar(other)),
        }
    }
}

impl From<Link> for Scalar {
    fn from(link: Link) -> Self {
        Value::from(link).into()
    }
}

impl From<TCRef> for Scalar {
    fn from(tc_ref: TCRef) -> Self {
        Self::Ref(Box::new(tc_ref))
    }
}

impl From<Value> for Scalar {
    fn from(value: Value) -> Scalar {
        Scalar::Value(value)
    }
}

impl TryFrom<Scalar> for Value {
    type Error = TCError;

    fn try_from(scalar: Scalar) -> TCResult<Value> {
        match scalar {
            Scalar::Value(value) => Ok(value),
            other => Err(TCError::bad_request("expected Value but found", other)),
        }
    }
}

impl TryCastFrom<Scalar> for TCRef {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Ref(_) => true,
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Ref(tc_ref) => Some(*tc_ref),
            _ => None,
        }
    }
}

impl TryCastFrom<Scalar> for IdRef {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Ref(tc_ref) => match &**tc_ref {
                TCRef::Id(_) => true,
                _ => false,
            },
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Ref(tc_ref) => match *tc_ref {
                TCRef::Id(id_ref) => Some(id_ref),
                _ => None,
            },
            _ => None,
        }
    }
}

impl TryCastFrom<Scalar> for Link {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Value(value) => Self::can_cast_from(value),
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Value(value) => Self::opt_cast_from(value),
            _ => None,
        }
    }
}

impl TryCastFrom<Scalar> for Number {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Value(value) => Self::can_cast_from(value),
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Value(value) => Self::opt_cast_from(value),
            _ => None,
        }
    }
}

impl<T: Clone + TryCastFrom<Scalar>> TryCastFrom<Scalar> for Map<T> {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Map(map) => HashMap::<Id, T>::can_cast_from(map),
            Scalar::Tuple(tuple) => Vec::<(Id, T)>::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Map(map) => HashMap::<Id, T>::opt_cast_from(map).map(Map::from),
            Scalar::Tuple(tuple) => {
                if let Some(entries) = Vec::<(Id, T)>::opt_cast_from(tuple) {
                    Some(entries.into_iter().collect())
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

impl<T: TryCastFrom<Scalar>> TryCastFrom<Scalar> for Vec<T> {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Tuple(tuple) => Self::opt_cast_from(tuple),
            _ => None,
        }
    }
}

impl TryCastFrom<Scalar> for Value {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Tuple(tuple) => tuple.iter().all(Self::can_cast_from),
            Scalar::Value(_) => true,
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Tuple(tuple) => {
                let mut value = Vec::with_capacity(tuple.len());
                for item in tuple.into_iter() {
                    if let Some(item) = Self::opt_cast_from(item) {
                        value.push(item);
                    } else {
                        return None;
                    }
                }
                Some(Value::Tuple(value.into()))
            }
            Scalar::Value(value) => Some(value),
            _ => None,
        }
    }
}

impl TryCastFrom<Scalar> for Id {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Value(value) => Self::can_cast_from(value),
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Value(value) => Self::opt_cast_from(value),
            _ => None,
        }
    }
}

impl<T: TryCastFrom<Scalar>> TryCastFrom<Scalar> for (T,) {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Tuple(tuple) => Self::opt_cast_from(tuple),
            _ => None,
        }
    }
}

impl<T1: TryCastFrom<Scalar>, T2: TryCastFrom<Scalar>> TryCastFrom<Scalar> for (T1, T2) {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Tuple(tuple) => Self::opt_cast_from(tuple),
            _ => None,
        }
    }
}

impl<T1: TryCastFrom<Scalar>, T2: TryCastFrom<Scalar>, T3: TryCastFrom<Scalar>> TryCastFrom<Scalar>
    for (T1, T2, T3)
{
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Tuple(tuple) => Self::opt_cast_from(tuple),
            _ => None,
        }
    }
}

#[derive(Default)]
pub struct ScalarVisitor {
    value: value::ValueVisitor,
}

impl ScalarVisitor {
    pub async fn visit_map_value<A: de::MapAccess>(
        class: ScalarType,
        access: &mut A,
    ) -> Result<Scalar, A::Error> {
        let scalar = access.next_value::<Scalar>(()).await?;

        if let Some(scalar) = scalar.clone().into_type(class) {
            return Ok(scalar);
        }

        let subject = Link::from(class.path()).into();
        let op_ref = if scalar.matches::<(Key, Scalar)>() {
            let (key, value) = scalar.opt_cast_into().unwrap();
            OpRef::Put((subject, key, value))
        } else if scalar.matches::<(Key,)>() {
            let (key,) = scalar.opt_cast_into().unwrap();
            OpRef::Get((subject, key))
        } else if scalar.matches::<Map<Scalar>>() {
            let params = scalar.opt_cast_into().unwrap();
            OpRef::Post((subject, params))
        } else {
            return Err(de::Error::invalid_type(
                scalar,
                format!("an Op with subject {}", subject),
            ));
        };

        Ok(TCRef::Op(op_ref).into())
    }

    pub fn visit_subject<E: de::Error>(subject: Subject, params: Scalar) -> Result<Scalar, E> {
        if params.is_none() {
            match subject {
                Subject::Ref(id, path) if path.is_empty() => {
                    Ok(Scalar::Ref(Box::new(TCRef::Id(id))))
                }
                Subject::Ref(id, path) => Ok(Scalar::Ref(Box::new(TCRef::Op(OpRef::Get((
                    Subject::Ref(id, path),
                    Value::default().into(),
                )))))),
                Subject::Link(link) => Ok(Scalar::Value(Value::Link(link))),
            }
        } else {
            OpRefVisitor::visit_ref_value(subject, params)
                .map(TCRef::Op)
                .map(Box::new)
                .map(Scalar::Ref)
        }
    }
}

#[async_trait]
impl de::Visitor for ScalarVisitor {
    type Value = Scalar;

    fn expecting() -> &'static str {
        "a Scalar, e.g. \"foo\" or 123 or {\"$ref: [\"id\", \"$state\"]\"}"
    }

    fn visit_i8<E: de::Error>(self, value: i8) -> Result<Self::Value, E> {
        self.value.visit_i8(value).map(Scalar::Value)
    }

    fn visit_i16<E: de::Error>(self, value: i16) -> Result<Self::Value, E> {
        self.value.visit_i16(value).map(Scalar::Value)
    }

    fn visit_i32<E: de::Error>(self, value: i32) -> Result<Self::Value, E> {
        self.value.visit_i32(value).map(Scalar::Value)
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
        self.value.visit_i64(value).map(Scalar::Value)
    }

    fn visit_u8<E: de::Error>(self, value: u8) -> Result<Self::Value, E> {
        self.value.visit_u8(value).map(Scalar::Value)
    }

    fn visit_u16<E: de::Error>(self, value: u16) -> Result<Self::Value, E> {
        self.value.visit_u16(value).map(Scalar::Value)
    }

    fn visit_u32<E: de::Error>(self, value: u32) -> Result<Self::Value, E> {
        self.value.visit_u32(value).map(Scalar::Value)
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
        self.value.visit_u64(value).map(Scalar::Value)
    }

    fn visit_f32<E: de::Error>(self, value: f32) -> Result<Self::Value, E> {
        self.value.visit_f32(value).map(Scalar::Value)
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
        self.value.visit_f64(value).map(Scalar::Value)
    }

    fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
        self.value.visit_string(value).map(Scalar::Value)
    }

    fn visit_byte_buf<E: de::Error>(self, buf: Vec<u8>) -> Result<Self::Value, E> {
        self.value.visit_byte_buf(buf).map(Scalar::Value)
    }

    fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
        self.value.visit_unit().map(Scalar::Value)
    }

    fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
        self.value.visit_none().map(Scalar::Value)
    }

    async fn visit_map<A: de::MapAccess>(self, mut access: A) -> Result<Self::Value, A::Error> {
        let key = if let Some(key) = access.next_key::<String>(()).await? {
            key
        } else {
            return Ok(Scalar::Map(Map::default()));
        };

        if let Ok(path) = TCPathBuf::from_str(&key) {
            if let Some(class) = ScalarType::from_path(&path) {
                if let Ok(scalar) = Self::visit_map_value(class, &mut access).await {
                    return Ok(scalar);
                }
            }
        }

        if let Ok(subject) = Subject::from_str(&key) {
            let params = access.next_value(()).await?;
            return Self::visit_subject(subject, params);
        }

        let mut map = HashMap::new();
        let key = Id::from_str(&key).map_err(de::Error::custom)?;
        let value = access.next_value(()).await?;
        map.insert(key, value);

        while let Some(key) = access.next_key(()).await? {
            let value = access.next_value(()).await?;
            map.insert(key, value);
        }

        Ok(Scalar::Map(map.into()))
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut access: A) -> Result<Self::Value, A::Error> {
        let mut items: Vec<Scalar> = if let Some(size) = access.size_hint() {
            Vec::with_capacity(size)
        } else {
            vec![]
        };

        while let Some(value) = access.next_element(()).await? {
            items.push(value)
        }

        Ok(Scalar::Tuple(items.into()))
    }
}

#[async_trait]
impl FromStream for Scalar {
    type Context = ();

    async fn from_stream<D: Decoder>(_: (), d: &mut D) -> Result<Self, D::Error> {
        d.decode_any(ScalarVisitor::default()).await
    }
}

impl<'en> ToStream<'en> for Scalar {
    fn to_stream<E: Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            Scalar::Map(map) => map.to_stream(e),
            Scalar::Op(op_def) => op_def.to_stream(e),
            Scalar::Ref(tc_ref) => tc_ref.to_stream(e),
            Scalar::Tuple(tuple) => tuple.to_stream(e),
            Scalar::Value(value) => value.to_stream(e),
        }
    }
}

impl<'en> IntoStream<'en> for Scalar {
    fn into_stream<E: Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            Scalar::Map(map) => map.into_inner().into_stream(e),
            Scalar::Op(op_def) => op_def.into_stream(e),
            Scalar::Ref(tc_ref) => tc_ref.into_stream(e),
            Scalar::Tuple(tuple) => tuple.into_inner().into_stream(e),
            Scalar::Value(value) => value.into_stream(e),
        }
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Scalar::Map(map) => fmt::Display::fmt(map, f),
            Scalar::Op(op) => fmt::Display::fmt(op, f),
            Scalar::Ref(tc_ref) => fmt::Display::fmt(tc_ref, f),
            Scalar::Tuple(tuple) => fmt::Display::fmt(tuple, f),
            Scalar::Value(value) => fmt::Display::fmt(value, f),
        }
    }
}
