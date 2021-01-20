use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use async_trait::async_trait;
use destream::de::*;
use destream::en::{Encoder, IntoStream, ToStream};
use futures::TryFutureExt;
use safecast::TryCastFrom;

use generic::*;

pub mod op;

pub use op::*;
pub use value::*;

const PREFIX: PathLabel = path_label(&["state", "scalar"]);

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum ScalarType {
    Map,
    Op(OpDefType),
    Tuple,
    Value(ValueType),
}

impl Class for ScalarType {
    type Instance = Scalar;
}

impl NativeClass for ScalarType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
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
            Self::Value(vt) => vt.path(),
            Self::Tuple => prefix.append(label("tuple")),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum Scalar {
    Map(Map<Self>),
    Op(OpDef),
    Tuple(Tuple<Self>),
    Value(Value),
}

impl Instance for Scalar {
    type Class = ScalarType;

    fn class(&self) -> ScalarType {
        use ScalarType as ST;
        match self {
            Self::Map(_) => ST::Map,
            Self::Op(op) => ST::Op(op.class()),
            Self::Tuple(_) => ST::Tuple,
            Self::Value(value) => ST::Value(value.class()),
        }
    }
}

struct ScalarVisitor {
    value_visitor: value::ValueVisitor,
}

#[async_trait]
impl Visitor for ScalarVisitor {
    type Value = Scalar;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a Tinychain Scalar, e.g. \"foo\" or 123 or {\"$ref: [\"id\", \"$state\"]\"}")
    }

    fn visit_i16<E: Error>(self, value: i16) -> Result<Self::Value, E> {
        self.value_visitor.visit_i16(value).map(Scalar::Value)
    }

    fn visit_i32<E: Error>(self, value: i32) -> Result<Self::Value, E> {
        self.value_visitor.visit_i32(value).map(Scalar::Value)
    }

    fn visit_i64<E: Error>(self, value: i64) -> Result<Self::Value, E> {
        self.value_visitor.visit_i64(value).map(Scalar::Value)
    }

    fn visit_u8<E: Error>(self, value: u8) -> Result<Self::Value, E> {
        self.value_visitor.visit_u8(value).map(Scalar::Value)
    }

    fn visit_u16<E: Error>(self, value: u16) -> Result<Self::Value, E> {
        self.value_visitor.visit_u16(value).map(Scalar::Value)
    }

    fn visit_u32<E: Error>(self, value: u32) -> Result<Self::Value, E> {
        self.value_visitor.visit_u32(value).map(Scalar::Value)
    }

    fn visit_u64<E: Error>(self, value: u64) -> Result<Self::Value, E> {
        self.value_visitor.visit_u64(value).map(Scalar::Value)
    }

    fn visit_f32<E: Error>(self, value: f32) -> Result<Self::Value, E> {
        self.value_visitor.visit_f32(value).map(Scalar::Value)
    }

    fn visit_f64<E: Error>(self, value: f64) -> Result<Self::Value, E> {
        self.value_visitor.visit_f64(value).map(Scalar::Value)
    }

    fn visit_string<E: Error>(self, value: String) -> Result<Self::Value, E> {
        self.value_visitor.visit_string(value).map(Scalar::Value)
    }

    async fn visit_map<A: MapAccess>(self, mut access: A) -> Result<Self::Value, A::Error> {
        if let Some(key) = access.next_key::<String>().await? {
            if let Ok(path) = TCPathBuf::from_str(&key) {
                if let Some(class) = ScalarType::from_path(&path) {
                    return match class {
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
        let value_visitor = value::ValueVisitor;
        d.decode_any(ScalarVisitor { value_visitor }).await
    }
}

impl<'en> ToStream<'en> for Scalar {
    fn to_stream<E: Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            Scalar::Map(map) => map.to_stream(e),
            Scalar::Op(op_def) => op_def.to_stream(e),
            Scalar::Tuple(tuple) => tuple.as_slice().into_stream(e),
            Scalar::Value(value) => value.to_stream(e),
        }
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Scalar::Map(map) => write!(
                f,
                "{{{}}}",
                map.iter()
                    .map(|(k, v)| format!("{}: {}", k, v))
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            Scalar::Op(op) => write!(f, "{}", op),
            Scalar::Tuple(tuple) => write!(
                f,
                "[{}]",
                tuple
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            Scalar::Value(value) => write!(f, "{}", value),
        }
    }
}
