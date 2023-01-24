//! A user-defined class.

use std::fmt;
use std::ops::Deref;

use async_hash::{Digest, Hash, Output};
use async_trait::async_trait;
use destream::{de, en};
use get_size::GetSize;
use get_size_derive::*;
use safecast::{CastFrom, TryCastFrom};
use tc_transact::fs::BlockData;

use tc_value::{Link, Value};
use tcgeneric::{path_label, Map, NativeClass, PathLabel};

use crate::scalar::*;
use crate::state::StateType;

use super::ObjectType;

const PATH: PathLabel = path_label(&["state", "class"]);

/// A user-defined class.
#[derive(Clone, Default, Eq, PartialEq, GetSize)]
pub struct InstanceClass {
    extends: Link,
    proto: Map<Scalar>,
}

impl InstanceClass {
    /// Construct a new base class.
    pub fn new(proto: Map<Scalar>) -> Self {
        Self {
            extends: PATH.into(),
            proto,
        }
    }

    /// Construct a new class which extends the class at `extends`.
    pub fn extend<L: Into<Link>>(extends: L, proto: Map<Scalar>) -> Self {
        Self {
            extends: extends.into(),
            proto,
        }
    }

    /// Borrow the link to this class' parent.
    pub fn extends(&self) -> &Link {
        &self.extends
    }

    /// Consume this class and return its data.
    pub fn into_inner(self) -> (Link, Map<Scalar>) {
        (self.extends, self.proto)
    }

    /// Borrow the prototype of this class.
    pub fn proto(&'_ self) -> &'_ Map<Scalar> {
        &self.proto
    }
}

impl<D: Digest> Hash<D> for InstanceClass {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash(&self)
    }
}

impl<'a, D: Digest> Hash<D> for &'a InstanceClass {
    fn hash(self) -> Output<D> {
        if self.extends == Link::from(PATH) {
            Hash::<D>::hash(self.proto.deref())
        } else {
            Hash::<D>::hash((&self.extends, self.proto.deref()))
        }
    }
}

impl tcgeneric::Class for InstanceClass {}

impl tcgeneric::Instance for InstanceClass {
    type Class = ObjectType;

    fn class(&self) -> ObjectType {
        ObjectType::Class
    }
}

impl BlockData for InstanceClass {
    fn ext() -> &'static str {
        "class"
    }
}

impl From<StateType> for InstanceClass {
    fn from(st: StateType) -> Self {
        Self::extend(st.path(), Map::default())
    }
}

impl CastFrom<Link> for InstanceClass {
    fn cast_from(extends: Link) -> Self {
        Self {
            extends,
            proto: Map::new(),
        }
    }
}

impl TryCastFrom<PostRef> for InstanceClass {
    fn can_cast_from(value: &PostRef) -> bool {
        match value {
            (Subject::Link(_), _) => true,
            _ => false,
        }
    }

    fn opt_cast_from(value: PostRef) -> Option<Self> {
        match value {
            (Subject::Link(extends), proto) => Some(Self { extends, proto }),
            _ => None,
        }
    }
}

impl TryCastFrom<OpRef> for InstanceClass {
    fn can_cast_from(op_ref: &OpRef) -> bool {
        match op_ref {
            OpRef::Post(op_ref) => Self::can_cast_from(op_ref),
            _ => false,
        }
    }

    fn opt_cast_from(op_ref: OpRef) -> Option<Self> {
        match op_ref {
            OpRef::Post(op_ref) => Self::opt_cast_from(op_ref),
            _ => None,
        }
    }
}

impl TryCastFrom<TCRef> for InstanceClass {
    fn can_cast_from(tc_ref: &TCRef) -> bool {
        match tc_ref {
            TCRef::Op(op_ref) => Self::can_cast_from(op_ref),
            _ => false,
        }
    }

    fn opt_cast_from(tc_ref: TCRef) -> Option<Self> {
        match tc_ref {
            TCRef::Op(op_ref) => Self::opt_cast_from(op_ref),
            _ => None,
        }
    }
}

impl TryCastFrom<Scalar> for InstanceClass {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Ref(tc_ref) => Self::can_cast_from(&**tc_ref),
            Scalar::Value(value) => Self::can_cast_from(value),
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Ref(tc_ref) => Self::opt_cast_from(*tc_ref),
            Scalar::Value(value) => Self::opt_cast_from(value),
            _ => None,
        }
    }
}

impl CastFrom<InstanceClass> for Scalar {
    fn cast_from(class: InstanceClass) -> Self {
        if class.proto.is_empty() {
            Self::Value(class.extends.into())
        } else {
            Self::Ref(Box::new(TCRef::Op(OpRef::Post((
                class.extends.into(),
                class.proto,
            )))))
        }
    }
}

impl TryCastFrom<Value> for InstanceClass {
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::Link(link) => Self::can_cast_from(link),
            _ => false,
        }
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        match value {
            Value::Link(link) => Self::opt_cast_from(link),
            _ => None,
        }
    }
}

impl TryCastFrom<InstanceClass> for StateType {
    fn can_cast_from(class: &InstanceClass) -> bool {
        if class.proto.is_empty() {
            if class.extends.host().is_none() {
                return StateType::from_path(class.extends.path()).is_some();
            }
        }

        false
    }

    fn opt_cast_from(class: InstanceClass) -> Option<Self> {
        if class.proto.is_empty() {
            if class.extends.host().is_none() {
                return StateType::from_path(class.extends.path());
            }
        }

        None
    }
}

impl TryCastFrom<InstanceClass> for Link {
    fn can_cast_from(class: &InstanceClass) -> bool {
        class.proto.is_empty()
    }

    fn opt_cast_from(class: InstanceClass) -> Option<Self> {
        if class.proto.is_empty() {
            Some(class.extends.into())
        } else {
            None
        }
    }
}

impl TryCastFrom<InstanceClass> for Value {
    fn can_cast_from(class: &InstanceClass) -> bool {
        Link::can_cast_from(class)
    }

    fn opt_cast_from(class: InstanceClass) -> Option<Self> {
        Link::opt_cast_from(class).map(Self::Link)
    }
}

#[async_trait]
impl de::FromStream for InstanceClass {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_map(InstanceClassVisitor).await
    }
}

impl<'en> en::ToStream<'en> for InstanceClass {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        use en::EncodeMap;

        let mut map = encoder.encode_map(Some(1))?;
        map.encode_entry(&self.extends, &self.proto)?;
        map.end()
    }
}

impl<'en> en::IntoStream<'en> for InstanceClass {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        use en::EncodeMap;

        let mut map = encoder.encode_map(Some(1))?;
        map.encode_entry(self.extends, self.proto)?;
        map.end()
    }
}

impl fmt::Debug for InstanceClass {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for InstanceClass {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Class")
    }
}

struct InstanceClassVisitor;

#[async_trait]
impl de::Visitor for InstanceClassVisitor {
    type Value = InstanceClass;

    fn expecting() -> &'static str {
        "a user-defined Class"
    }

    async fn visit_map<A: de::MapAccess>(self, mut access: A) -> Result<InstanceClass, A::Error> {
        if let Some(extends) = access.next_key::<Link>(()).await? {
            log::debug!("Class extends {}", extends);
            let proto = access.next_value(()).await?;
            log::debug!("prototype is {}", proto);
            return Ok(InstanceClass::extend(extends, proto));
        } else {
            Err(de::Error::invalid_length(0, 1))
        }
    }
}
