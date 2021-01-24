use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use async_trait::async_trait;
use destream::de::*;
use destream::en::{Encoder, IntoStream, ToStream};
use futures::TryFutureExt;
use log::debug;
use safecast::{Match, TryCastFrom};

use generic::*;

pub mod op;
pub mod reference;

pub use op::*;
pub use reference::*;
pub use value::*;

const PREFIX: PathLabel = path_label(&["state", "scalar"]);

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

#[derive(Clone, Eq, PartialEq)]
pub enum Scalar {
    Map(Map<Self>),
    Op(OpDef),
    Ref(TCRef),
    Tuple(Tuple<Self>),
    Value(Value),
}

impl Scalar {
    pub fn is_none(&self) -> bool {
        match self {
            Self::Tuple(tuple) => tuple.is_empty(),
            Self::Value(value) => value.is_none(),
            _ => false,
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

impl From<Value> for Scalar {
    fn from(value: Value) -> Scalar {
        Scalar::Value(value)
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

#[derive(Default)]
pub struct ScalarVisitor {
    value: value::ValueVisitor,
}

impl ScalarVisitor {
    pub async fn visit_map_value<A: MapAccess>(
        class: ScalarType,
        access: &mut A,
    ) -> Result<Scalar, A::Error> {
        match class {
            ScalarType::Map => {
                access
                    .next_value::<HashMap<Id, Scalar>>()
                    .map_ok(Map::from)
                    .map_ok(Scalar::Map)
                    .await
            }
            ScalarType::Op(odt) => {
                OpDefVisitor::visit_map_value(odt, access)
                    .map_ok(Scalar::Op)
                    .await
            }
            ScalarType::Ref(_rt) => {
                unimplemented!()
            }
            ScalarType::Tuple => {
                access
                    .next_value::<Vec<Scalar>>()
                    .map_ok(Tuple::from)
                    .map_ok(Scalar::Tuple)
                    .await
            }
            ScalarType::Value(vt) => {
                ValueVisitor::visit_map_value_async(vt, access)
                    .map_ok(Scalar::Value)
                    .await
            }
        }
    }
}

#[async_trait]
impl Visitor for ScalarVisitor {
    type Value = Scalar;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a Tinychain Scalar, e.g. \"foo\" or 123 or {\"$ref: [\"id\", \"$state\"]\"}")
    }

    fn visit_i8<E: Error>(self, value: i8) -> Result<Self::Value, E> {
        self.value.visit_i8(value).map(Scalar::Value)
    }

    fn visit_i16<E: Error>(self, value: i16) -> Result<Self::Value, E> {
        self.value.visit_i16(value).map(Scalar::Value)
    }

    fn visit_i32<E: Error>(self, value: i32) -> Result<Self::Value, E> {
        self.value.visit_i32(value).map(Scalar::Value)
    }

    fn visit_i64<E: Error>(self, value: i64) -> Result<Self::Value, E> {
        self.value.visit_i64(value).map(Scalar::Value)
    }

    fn visit_u8<E: Error>(self, value: u8) -> Result<Self::Value, E> {
        self.value.visit_u8(value).map(Scalar::Value)
    }

    fn visit_u16<E: Error>(self, value: u16) -> Result<Self::Value, E> {
        self.value.visit_u16(value).map(Scalar::Value)
    }

    fn visit_u32<E: Error>(self, value: u32) -> Result<Self::Value, E> {
        self.value.visit_u32(value).map(Scalar::Value)
    }

    fn visit_u64<E: Error>(self, value: u64) -> Result<Self::Value, E> {
        self.value.visit_u64(value).map(Scalar::Value)
    }

    fn visit_f32<E: Error>(self, value: f32) -> Result<Self::Value, E> {
        self.value.visit_f32(value).map(Scalar::Value)
    }

    fn visit_f64<E: Error>(self, value: f64) -> Result<Self::Value, E> {
        self.value.visit_f64(value).map(Scalar::Value)
    }

    fn visit_string<E: Error>(self, value: String) -> Result<Self::Value, E> {
        self.value.visit_string(value).map(Scalar::Value)
    }

    fn visit_byte_buf<E: Error>(self, buf: Vec<u8>) -> Result<Self::Value, E> {
        self.value.visit_byte_buf(buf).map(Scalar::Value)
    }

    fn visit_unit<E: Error>(self) -> Result<Self::Value, E> {
        self.value.visit_unit().map(Scalar::Value)
    }

    fn visit_none<E: Error>(self) -> Result<Self::Value, E> {
        self.value.visit_none().map(Scalar::Value)
    }

    async fn visit_map<A: MapAccess>(self, mut access: A) -> Result<Self::Value, A::Error> {
        if let Some(key) = access.next_key::<String>().await? {
            if let Ok(link) = Link::from_str(&key) {
                if link.host().is_none() {
                    if let Some(class) = ScalarType::from_path(link.path()) {
                        return Self::visit_map_value(class, &mut access).await;
                    }
                } else {
                    let params: Scalar = access.next_value().await?;
                    return if params.is_none() {
                        Ok(Value::Link(link).into())
                    } else if params.matches::<(Value, Scalar)>() {
                        unimplemented!()
                    } else if params.matches::<(Value,)>() {
                        unimplemented!()
                    } else if let Scalar::Map(_params) = params {
                        unimplemented!()
                    } else {
                        Err(destream::de::Error::invalid_type(
                            params,
                            &"a Link or OpRef",
                        ))
                    };
                }
            }

            let key = Id::try_cast_from(key, |id| format!("invalid Id: {}", id))
                .map_err(A::Error::custom)?;

            let mut map = HashMap::new();
            let value = access.next_value().await?;
            map.insert(key, value);

            while let Some(key) = access.next_key().await? {
                let value = access.next_value().await?;
                map.insert(key, value);
            }

            Ok(Scalar::Map(map.into()))
        } else {
            Ok(Scalar::Map(Map::default()))
        }
    }

    async fn visit_seq<A: SeqAccess>(self, mut access: A) -> Result<Self::Value, A::Error> {
        let mut items: Vec<Scalar> = if let Some(size) = access.size_hint() {
            Vec::with_capacity(size)
        } else {
            vec![]
        };

        while let Some(value) = access.next_element().await? {
            items.push(value)
        }

        Ok(Scalar::Tuple(items.into()))
    }
}

#[async_trait]
impl FromStream for Scalar {
    async fn from_stream<D: Decoder>(d: &mut D) -> Result<Self, D::Error> {
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
