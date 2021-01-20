use std::fmt;

use async_trait::async_trait;
use destream::{de, Decoder, Encoder, FromStream, IntoStream, ToStream};
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

impl TryCastFrom<Scalar> for Id {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Value(value) => Id::can_cast_from(value),
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Id> {
        match scalar {
            Scalar::Value(value) => Id::opt_cast_from(value),
            _ => None,
        }
    }
}

impl<T: TryCastFrom<Scalar>> TryCastFrom<Scalar> for Vec<T> {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Tuple(t) => Self::can_cast_from(t),
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Vec<T>> {
        match scalar {
            Scalar::Tuple(t) => Self::opt_cast_from(t),
            _ => None,
        }
    }
}

impl<T: TryCastFrom<Scalar>> TryCastFrom<Scalar> for (T,) {
    fn can_cast_from(source: &Scalar) -> bool {
        if let Scalar::Tuple(source) = source {
            Self::can_cast_from(source)
        } else {
            false
        }
    }

    fn opt_cast_from(source: Scalar) -> Option<(T,)> {
        if let Scalar::Tuple(source) = source {
            Self::opt_cast_from(source)
        } else {
            None
        }
    }
}

impl<T1: TryCastFrom<Scalar>, T2: TryCastFrom<Scalar>> TryCastFrom<Scalar> for (T1, T2) {
    fn can_cast_from(source: &Scalar) -> bool {
        if let Scalar::Tuple(source) = source {
            Self::can_cast_from(source)
        } else {
            false
        }
    }

    fn opt_cast_from(source: Scalar) -> Option<(T1, T2)> {
        if let Scalar::Tuple(source) = source {
            Self::opt_cast_from(source)
        } else {
            None
        }
    }
}

impl<T1: TryCastFrom<Scalar>, T2: TryCastFrom<Scalar>, T3: TryCastFrom<Scalar>> TryCastFrom<Scalar>
    for (T1, T2, T3)
{
    fn can_cast_from(source: &Scalar) -> bool {
        if let Scalar::Tuple(source) = source {
            Self::can_cast_from(source)
        } else {
            false
        }
    }

    fn opt_cast_from(source: Scalar) -> Option<(T1, T2, T3)> {
        if let Scalar::Tuple(source) = source {
            Self::opt_cast_from(source)
        } else {
            None
        }
    }
}

impl<
        T1: TryCastFrom<Scalar>,
        T2: TryCastFrom<Scalar>,
        T3: TryCastFrom<Scalar>,
        T4: TryCastFrom<Scalar>,
    > TryCastFrom<Scalar> for (T1, T2, T3, T4)
{
    fn can_cast_from(source: &Scalar) -> bool {
        if let Scalar::Tuple(source) = source {
            Self::can_cast_from(source)
        } else {
            false
        }
    }

    fn opt_cast_from(source: Scalar) -> Option<(T1, T2, T3, T4)> {
        if let Scalar::Tuple(source) = source {
            Self::opt_cast_from(source)
        } else {
            None
        }
    }
}

struct ScalarVisitor {
    value_visitor: value::ValueVisitor,
}

#[async_trait]
impl de::Visitor for ScalarVisitor {
    type Value = Scalar;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a Tinychain Scalar, e.g. \"foo\" or 123 or {\"$ref: [\"id\", \"$state\"]\"}")
    }

    fn visit_i16<E: de::Error>(self, value: i16) -> Result<Self::Value, E> {
        self.value_visitor.visit_i16(value).map(Scalar::Value)
    }

    fn visit_i32<E: de::Error>(self, value: i32) -> Result<Self::Value, E> {
        self.value_visitor.visit_i32(value).map(Scalar::Value)
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
        self.value_visitor.visit_i64(value).map(Scalar::Value)
    }

    fn visit_u8<E: de::Error>(self, value: u8) -> Result<Self::Value, E> {
        self.value_visitor.visit_u8(value).map(Scalar::Value)
    }

    fn visit_u16<E: de::Error>(self, value: u16) -> Result<Self::Value, E> {
        self.value_visitor.visit_u16(value).map(Scalar::Value)
    }

    fn visit_u32<E: de::Error>(self, value: u32) -> Result<Self::Value, E> {
        self.value_visitor.visit_u32(value).map(Scalar::Value)
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
        self.value_visitor.visit_u64(value).map(Scalar::Value)
    }

    fn visit_f32<E: de::Error>(self, value: f32) -> Result<Self::Value, E> {
        self.value_visitor.visit_f32(value).map(Scalar::Value)
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
        self.value_visitor.visit_f64(value).map(Scalar::Value)
    }

    fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
        self.value_visitor.visit_string(value).map(Scalar::Value)
    }

    async fn visit_map<A: de::MapAccess>(self, _access: A) -> Result<Self::Value, A::Error> {
        unimplemented!()
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut access: A) -> Result<Self::Value, A::Error> {
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
