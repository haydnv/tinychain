use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::str::FromStr;

use log::debug;
use serde::de;
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};

use crate::class::*;
use crate::error;

pub mod object;
pub mod op;
pub mod value;

pub use op::*;
pub use value::*;

pub type Object = object::Object;

pub trait CastFrom<T> {
    fn cast_from(value: T) -> Self;
}

pub trait CastInto<T> {
    fn cast_into(self) -> T;
}

impl<T> CastFrom<T> for T {
    fn cast_from(value: T) -> Self {
        value
    }
}

impl<T, F: CastFrom<T>> CastInto<F> for T {
    fn cast_into(self) -> F {
        F::cast_from(self)
    }
}

pub trait TryCastFrom<T>: Sized {
    fn can_cast_from(value: &T) -> bool;

    fn opt_cast_from(value: T) -> Option<Self>;

    fn try_cast_from<Err, E: FnOnce(&T) -> Err>(value: T, on_err: E) -> Result<Self, Err> {
        if Self::can_cast_from(&value) {
            Ok(Self::opt_cast_from(value).unwrap())
        } else {
            Err(on_err(&value))
        }
    }
}

pub trait TryCastInto<T>: Sized {
    fn can_cast_into(&self) -> bool;

    fn opt_cast_into(self) -> Option<T>;

    fn try_cast_into<Err, E: FnOnce(&Self) -> Err>(self, on_err: E) -> Result<T, Err> {
        if self.can_cast_into() {
            Ok(self.opt_cast_into().unwrap())
        } else {
            Err(on_err(&self))
        }
    }
}

impl<F, T: CastFrom<F>> TryCastFrom<F> for T {
    fn can_cast_from(_: &F) -> bool {
        true
    }

    fn opt_cast_from(f: F) -> Option<T> {
        Some(T::cast_from(f))
    }
}

impl<F, T: TryCastFrom<F>> TryCastInto<T> for F {
    fn can_cast_into(&self) -> bool {
        T::can_cast_from(self)
    }

    fn opt_cast_into(self) -> Option<T> {
        T::opt_cast_from(self)
    }
}

pub trait ScalarInstance: Instance + Sized {
    type Class: ScalarClass;

    fn matches<T: TryCastFrom<Self>>(&self) -> bool {
        T::can_cast_from(self)
    }
}

pub trait ScalarClass: Class {
    type Instance: ScalarInstance;

    fn try_cast<S: Into<Scalar>>(&self, scalar: S) -> TCResult<<Self as ScalarClass>::Instance>;
}

#[derive(Clone, Eq, PartialEq)]
pub enum ScalarType {
    Object,
    Op(op::OpType),
    Value(ValueType),
    Tuple,
}

impl Class for ScalarType {
    type Instance = Scalar;
}

impl NativeClass for ScalarType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.is_empty() {
            return Err(error::method_not_allowed(TCPath::from(path)));
        }

        match suffix[0].as_str() {
            "object" if suffix.len() == 1 => Ok(ScalarType::Object),
            "op" => op::OpType::from_path(path).map(ScalarType::Op),
            "value" => ValueType::from_path(path).map(ScalarType::Value),
            "tuple" if suffix.len() == 1 => Ok(ScalarType::Tuple),
            other => Err(error::not_found(other)),
        }
    }

    fn prefix() -> TCPathBuf {
        TCType::prefix()
    }
}

impl ScalarClass for ScalarType {
    type Instance = Scalar;

    fn try_cast<S: Into<Scalar>>(&self, scalar: S) -> TCResult<Scalar> {
        let scalar: Scalar = scalar.into();

        match self {
            Self::Object => match scalar {
                Scalar::Object(map) => Ok(Scalar::Object(map)),
                other => Err(error::bad_request("Cannot cast into Map from", other)),
            },
            Self::Op(ot) => ot.try_cast(scalar).map(Box::new).map(Scalar::Op),
            Self::Value(vt) => vt.try_cast(scalar).map(Scalar::Value),
            Self::Tuple => scalar
                .try_cast_into(|v| error::not_implemented(format!("Cast into Tuple from {}", v))),
        }
    }
}

impl From<ScalarType> for Link {
    fn from(st: ScalarType) -> Link {
        match st {
            ScalarType::Object => ScalarType::prefix().append(label("object")).into(),
            ScalarType::Tuple => ScalarType::prefix().append(label("tuple")).into(),
            ScalarType::Op(ot) => ot.into(),
            ScalarType::Value(vt) => vt.into(),
        }
    }
}

impl From<ScalarType> for TCType {
    fn from(st: ScalarType) -> TCType {
        TCType::Scalar(st)
    }
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Object => write!(f, "type Object (generic)"),
            Self::Op(ot) => write!(f, "{}", ot),
            Self::Value(vt) => write!(f, "{}", vt),
            Self::Tuple => write!(f, "type Tuple"),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum Scalar {
    Object(Object),
    Op(Box<op::Op>),
    Value(value::Value),
    Tuple(Vec<Scalar>),
}

impl Instance for Scalar {
    type Class = ScalarType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Object(_) => ScalarType::Object,
            Self::Op(op) => ScalarType::Op(op.class()),
            Self::Value(value) => ScalarType::Value(value.class()),
            Self::Tuple(_) => ScalarType::Tuple,
        }
    }
}

impl ScalarInstance for Scalar {
    type Class = ScalarType;
}

impl From<Number> for Scalar {
    fn from(n: Number) -> Scalar {
        Scalar::Value(Value::Number(n))
    }
}

impl From<Object> for Scalar {
    fn from(object: Object) -> Scalar {
        Scalar::Object(object)
    }
}

impl From<Op> for Scalar {
    fn from(op: Op) -> Scalar {
        Scalar::Op(Box::new(op))
    }
}

impl From<Value> for Scalar {
    fn from(value: Value) -> Scalar {
        Scalar::Value(value)
    }
}

impl From<ValueId> for Scalar {
    fn from(id: ValueId) -> Scalar {
        Scalar::Value(id.into())
    }
}

impl<T1: Into<Scalar>, T2: Into<Scalar>> From<(T1, T2)> for Scalar {
    fn from(tuple: (T1, T2)) -> Scalar {
        Scalar::Tuple(vec![tuple.0.into(), tuple.1.into()])
    }
}

impl<T: Into<Scalar>> From<Vec<T>> for Scalar {
    fn from(mut v: Vec<T>) -> Scalar {
        Scalar::Tuple(v.drain(..).map(|i| i.into()).collect())
    }
}

impl PartialEq<Value> for Scalar {
    fn eq(&self, that: &Value) -> bool {
        match self {
            Self::Value(this) => this == that,
            _ => false,
        }
    }
}

impl TryFrom<Scalar> for object::Object {
    type Error = error::TCError;

    fn try_from(s: Scalar) -> TCResult<object::Object> {
        match s {
            Scalar::Object(object) => Ok(object),
            other => Err(error::bad_request(
                "Expected generic Object but found",
                other,
            )),
        }
    }
}

impl TryFrom<Scalar> for Value {
    type Error = error::TCError;

    fn try_from(s: Scalar) -> TCResult<Value> {
        match s {
            Scalar::Value(value) => Ok(value),
            other => Err(error::bad_request("Expected Value but found", other)),
        }
    }
}

impl TryFrom<Scalar> for Vec<Scalar> {
    type Error = error::TCError;

    fn try_from(s: Scalar) -> TCResult<Vec<Scalar>> {
        match s {
            Scalar::Tuple(t) => Ok(t),
            other => Err(error::bad_request("Expected Tuple, found", other)),
        }
    }
}

impl<T: TryFrom<Scalar, Error = error::TCError>> TryFrom<Scalar> for Vec<T> {
    type Error = error::TCError;

    fn try_from(source: Scalar) -> TCResult<Vec<T>> {
        let mut source: Vec<Scalar> = source.try_into()?;
        let mut items = Vec::with_capacity(source.len());
        for item in source.drain(..) {
            items.push(item.try_into()?);
        }
        Ok(items)
    }
}

impl TryCastFrom<Scalar> for Object {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Object(_) => true,
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Object> {
        match scalar {
            Scalar::Object(object) => Some(object),
            _ => None,
        }
    }
}

impl TryCastFrom<Scalar> for Value {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Value(_) => true,
            Scalar::Tuple(tuple) => Value::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Value> {
        debug!("cast into Value from {}", scalar);

        match scalar {
            Scalar::Value(value) => Some(value),
            Scalar::Tuple(tuple) => Value::opt_cast_from(tuple),
            _ => None,
        }
    }
}

impl TryCastFrom<Scalar> for Link {
    fn can_cast_from(scalar: &Scalar) -> bool {
        if let Scalar::Value(value) = scalar {
            Link::can_cast_from(value)
        } else {
            false
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Link> {
        if let Scalar::Value(value) = scalar {
            Link::opt_cast_from(value)
        } else {
            None
        }
    }
}

impl TryCastFrom<Scalar> for TCRef {
    fn can_cast_from(scalar: &Scalar) -> bool {
        if let Scalar::Value(value) = scalar {
            TCRef::can_cast_from(value)
        } else {
            false
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<TCRef> {
        if let Scalar::Value(value) = scalar {
            TCRef::opt_cast_from(value)
        } else {
            None
        }
    }
}

impl TryCastFrom<Scalar> for ValueId {
    fn can_cast_from(scalar: &Scalar) -> bool {
        if let Scalar::Value(value) = scalar {
            ValueId::can_cast_from(value)
        } else {
            false
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<ValueId> {
        if let Scalar::Value(value) = scalar {
            ValueId::opt_cast_from(value)
        } else {
            None
        }
    }
}

impl<T: TryCastFrom<Scalar>> TryCastFrom<Scalar> for Vec<T> {
    fn can_cast_from(scalar: &Scalar) -> bool {
        if let Scalar::Tuple(values) = scalar {
            values.iter().all(T::can_cast_from)
        } else {
            false
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Vec<T>> {
        if let Scalar::Tuple(mut values) = scalar {
            let mut cast: Vec<T> = Vec::with_capacity(values.len());
            for val in values.drain(..) {
                if let Some(val) = val.opt_cast_into() {
                    cast.push(val)
                } else {
                    return None;
                }
            }

            Some(cast)
        } else {
            None
        }
    }
}

impl<T: TryCastFrom<Scalar>> TryCastFrom<Scalar> for (T,) {
    fn can_cast_from(source: &Scalar) -> bool {
        if let Scalar::Tuple(source) = source {
            if source.len() == 1 && T::can_cast_from(&source[0]) {
                return true;
            }
        }

        false
    }

    fn opt_cast_from(source: Scalar) -> Option<(T,)> {
        if let Scalar::Tuple(mut source) = source {
            if source.len() == 1 {
                return source.pop().unwrap().opt_cast_into().map(|item| (item,));
            }
        }

        None
    }
}

impl<T1: TryCastFrom<Scalar>, T2: TryCastFrom<Scalar>> TryCastFrom<Scalar> for (T1, T2) {
    fn can_cast_from(source: &Scalar) -> bool {
        if let Scalar::Tuple(source) = source {
            if source.len() == 2 && T1::can_cast_from(&source[0]) && T2::can_cast_from(&source[1]) {
                return true;
            }
        }

        false
    }

    fn opt_cast_from(source: Scalar) -> Option<(T1, T2)> {
        if let Scalar::Tuple(mut source) = source {
            if source.len() == 2 {
                let second: Option<T2> = source.pop().unwrap().opt_cast_into();
                let first: Option<T1> = source.pop().unwrap().opt_cast_into();
                return match (first, second) {
                    (Some(first), Some(second)) => Some((first, second)),
                    _ => None,
                };
            }
        }

        None
    }
}

impl<T1: TryCastFrom<Scalar>, T2: TryCastFrom<Scalar>, T3: TryCastFrom<Scalar>> TryCastFrom<Scalar>
    for (T1, T2, T3)
{
    fn can_cast_from(source: &Scalar) -> bool {
        if let Scalar::Tuple(source) = source {
            if source.len() == 3
                && T1::can_cast_from(&source[0])
                && T2::can_cast_from(&source[1])
                && T3::can_cast_from(&source[2])
            {
                return true;
            }
        }

        false
    }

    fn opt_cast_from(source: Scalar) -> Option<(T1, T2, T3)> {
        if let Scalar::Tuple(mut source) = source {
            if source.len() == 3 {
                let third: Option<T3> = source.pop().unwrap().opt_cast_into();
                let second: Option<T2> = source.pop().unwrap().opt_cast_into();
                let first: Option<T1> = source.pop().unwrap().opt_cast_into();
                return match (first, second, third) {
                    (Some(first), Some(second), Some(third)) => Some((first, second, third)),
                    _ => None,
                };
            }
        }

        None
    }
}

struct ScalarVisitor {
    value_visitor: value::ValueVisitor,
}

impl ScalarVisitor {
    fn visit_op_ref<E>(self, link: Link, params: Scalar) -> Result<Scalar, E>
    where
        E: de::Error,
    {
        let op_ref = if params.matches::<(Value, Scalar)>() {
            let (key, value) = params.opt_cast_into().unwrap();
            OpRef::Put((link, key, value))
        } else if params.matches::<(Value,)>() {
            debug!("GET ref, key is {}", params);
            OpRef::Get((link, params.opt_cast_into().unwrap()))
        } else if params.matches::<Object>() {
            OpRef::Post((link, params.opt_cast_into().unwrap()))
        } else {
            return Err(de::Error::custom(format!("invalid Op format: {}", params)));
        };

        Ok(Scalar::Op(Box::new(Op::Ref(op_ref))))
    }
}

impl<'de> de::Visitor<'de> for ScalarVisitor {
    type Value = Scalar;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(
            "a Tinychain Scalar, e.g. \"foo\" or 123 or {\"$ref: [\"slice_id\", \"$state\"]\"}",
        )
    }

    fn visit_f32<E>(self, value: f32) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.value_visitor.visit_f32(value).map(Scalar::Value)
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.value_visitor.visit_f64(value).map(Scalar::Value)
    }

    fn visit_i16<E>(self, value: i16) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.value_visitor.visit_i16(value).map(Scalar::Value)
    }

    fn visit_i32<E>(self, value: i32) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.value_visitor.visit_i32(value).map(Scalar::Value)
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.value_visitor.visit_i64(value).map(Scalar::Value)
    }

    fn visit_u8<E>(self, value: u8) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.value_visitor.visit_u8(value).map(Scalar::Value)
    }

    fn visit_u16<E>(self, value: u16) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.value_visitor.visit_u16(value).map(Scalar::Value)
    }

    fn visit_u32<E>(self, value: u32) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.value_visitor.visit_u32(value).map(Scalar::Value)
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.value_visitor.visit_u64(value).map(Scalar::Value)
    }

    fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
    where
        M: de::MapAccess<'de>,
    {
        let mut data: HashMap<String, Scalar> = HashMap::new();

        while let Some(key) = access.next_key()? {
            match access.next_value()? {
                Some(value) => {
                    data.insert(key, value);
                }
                None => {
                    return Err(de::Error::custom(format!(
                        "Failed to parse value of {}",
                        key
                    )))
                }
            }
        }

        if data.is_empty() {
            return Ok(Scalar::Object(Object::default()));
        } else if data.len() == 1 {
            debug!("deserialize map of length 1");
            let (key, data) = data.clone().drain().next().unwrap();

            if key.starts_with('$') {
                debug!("key is a Ref: {}", key);

                let (subject, path) = if let Some(i) = key.find('/') {
                    let (subject, path) = key.split_at(i);
                    let subject = TCRef::from_str(subject).map_err(de::Error::custom)?;
                    let path = TCPathBuf::from_str(path).map_err(de::Error::custom)?;
                    (subject, path)
                } else {
                    (
                        TCRef::from_str(&key).map_err(de::Error::custom)?,
                        TCPathBuf::default(),
                    )
                };

                return if data == Scalar::Tuple(vec![]) || data == Scalar::Value(Value::None) {
                    if path == TCPathBuf::default() {
                        Ok(Scalar::Value(subject.into()))
                    } else {
                        Ok(Scalar::Op(Box::new(Op::Method(Method::Get(
                            (subject, path),
                            Value::None,
                        )))))
                    }
                } else {
                    debug!("map is a Method with parameters {}", data);

                    let method = if data.matches::<Object>() {
                        Method::Post((subject, path), data.opt_cast_into().unwrap())
                    } else {
                        let mut data: Vec<Scalar> = data.try_into().map_err(de::Error::custom)?;
                        if data.len() == 1 {
                            let key = data.pop().unwrap().try_into().map_err(de::Error::custom)?;
                            Method::Get((subject, path), key)
                        } else if data.len() == 2 {
                            let value = data.pop().unwrap();
                            let key = data.pop().unwrap();
                            debug!("key is {}", key);
                            let key = key.try_cast_into(|v| {
                                de::Error::custom(format!("Method key must be a Value, not {}", v))
                            })?;
                            Method::Put((subject, path), (key, value))
                        } else {
                            return Err(de::Error::custom(format!(
                                "Expected a Method but found: {}",
                                Scalar::Tuple(data)
                            )));
                        }
                    };

                    Ok(Scalar::Op(Box::new(Op::Method(method))))
                };
            } else if let Ok(link) = key.parse::<link::Link>() {
                debug!("key is a Link: {}", link);

                if data == Value::None
                    || data == Value::Tuple(vec![])
                    || data == Scalar::Tuple(vec![])
                {
                    return Ok(Scalar::Value(Value::TCString(link.into())));
                }

                let path = link.path();
                return if path.len() > 1 && &path[0] == "sbin" {
                    match path[1].as_str() {
                        "value" | "op" | "tuple" | "object" => match data {
                            Scalar::Value(data) => match data {
                                Value::Tuple(tuple) if &path[1] == "tuple" => {
                                    Ok(Scalar::Value(Value::Tuple(tuple)))
                                }
                                Value::Tuple(mut tuple) if tuple.len() == 1 => {
                                    let key = tuple.pop().unwrap();
                                    let dtype = ValueType::from_path(&link.path()[..])
                                        .map_err(de::Error::custom)?;

                                    dtype
                                        .try_cast(key)
                                        .map(Scalar::Value)
                                        .map_err(de::Error::custom)
                                }
                                key => {
                                    let dtype = ValueType::from_path(&link.path()[..])
                                        .map_err(de::Error::custom)?;

                                    dtype
                                        .try_cast(key)
                                        .map(Scalar::Value)
                                        .map_err(de::Error::custom)
                                }
                            },
                            Scalar::Op(_) => {
                                Err(de::Error::custom("{<link>: <op>} does not make sense"))
                            }
                            Scalar::Object(object) if path.len() == 2 && &path[1] == "object" => {
                                Ok(Scalar::Object(object))
                            }
                            Scalar::Object(object) => {
                                Ok(Scalar::Op(Box::new(Op::Ref(OpRef::Post((link, object))))))
                            }
                            other => {
                                let dtype = ScalarType::from_path(&link.path()[..])
                                    .map_err(de::Error::custom)?;
                                dtype.try_cast(other).map_err(de::Error::custom)
                            }
                        },
                        _ if data == Value::None || data == Value::Tuple(vec![]) => {
                            Ok(Scalar::Value(Value::TCString(link.into())))
                        }
                        _ => self.visit_op_ref::<M::Error>(link, data),
                    }
                } else {
                    self.visit_op_ref::<M::Error>(link, data)
                };
            }
        }

        debug!("map is an Object");

        let mut map = HashMap::with_capacity(data.len());
        for (key, value) in data.drain() {
            debug!("key {} value {}", key, value);
            let key: ValueId = key.parse().map_err(de::Error::custom)?;
            map.insert(key, value);
        }

        Ok(Scalar::Object(map.into()))
    }

    fn visit_seq<L>(self, mut access: L) -> Result<Self::Value, L::Error>
    where
        L: de::SeqAccess<'de>,
    {
        let mut items: Vec<Scalar> = if let Some(size) = access.size_hint() {
            Vec::with_capacity(size)
        } else {
            vec![]
        };

        while let Some(value) = access.next_element()? {
            items.push(value)
        }

        Ok(Scalar::Tuple(items))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.value_visitor.visit_str(value).map(Scalar::Value)
    }
}

impl<'de> de::Deserialize<'de> for Scalar {
    fn deserialize<D: de::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let value_visitor = value::ValueVisitor;
        d.deserialize_any(ScalarVisitor { value_visitor })
    }
}

impl Serialize for Scalar {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Scalar::Object(object) => object.serialize(s),
            Scalar::Op(op) => match &**op {
                Op::Def(def) => {
                    let mut map = s.serialize_map(Some(1))?;
                    map.serialize_entry(
                        &Link::from(def.class()).to_string(),
                        &Scalar::cast_from(def.clone()),
                    )?;
                    map.end()
                }
                Op::Method(method) => {
                    let ((subject, path), args): ((TCRef, TCPathBuf), Scalar) = match method {
                        Method::Get(subject, arg) => (subject.clone(), vec![arg.clone()].into()),
                        Method::Put(subject, args) => (subject.clone(), args.clone().into()),
                        Method::Post(subject, args) => (subject.clone(), args.clone().into()),
                    };

                    let mut map = s.serialize_map(Some(1))?;
                    map.serialize_entry(&format!("{}{}", subject, path), &args)?;
                    map.end()
                }
                Op::Ref(op_ref) => match op_ref {
                    OpRef::If(cond, then, or_else) => {
                        let mut map = s.serialize_map(Some(1))?;
                        map.serialize_entry(
                            &Link::from(op_ref.class()).to_string(),
                            &[&Scalar::Value(cond.clone().into()), then, or_else],
                        )?;
                        map.end()
                    }
                    OpRef::Get((path, key)) => {
                        let mut map = s.serialize_map(Some(1))?;
                        map.serialize_entry(&path.to_string(), key)?;
                        map.end()
                    }
                    OpRef::Put((path, key, value)) => {
                        let mut map = s.serialize_map(Some(1))?;
                        map.serialize_entry(
                            &path.to_string(),
                            &[&Scalar::Value(key.clone()), value],
                        )?;
                        map.end()
                    }
                    OpRef::Post((path, data)) => {
                        let mut map = s.serialize_map(Some(1))?;
                        map.serialize_entry(&path.to_string(), data)?;
                        map.end()
                    }
                },
            },
            Scalar::Value(value) => value.serialize(s),
            Scalar::Tuple(tuple) => {
                let mut seq = s.serialize_seq(Some(tuple.len()))?;
                for item in tuple {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
        }
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Scalar::Object(object) => write!(
                f,
                "{{{}}}",
                object
                    .data()
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, v))
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            Scalar::Op(op) => write!(f, "{}", op),
            Scalar::Value(value) => write!(f, "{}", value),
            Scalar::Tuple(tuple) => write!(
                f,
                "[{}]",
                tuple
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
        }
    }
}
