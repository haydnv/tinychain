//! User-defined public-orientation features.

use std::fmt;
use std::marker::PhantomData;

use async_trait::async_trait;
use destream::{de, en};
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tc_scalar::Scalar;
use tc_transact::hash::{AsyncHash, Hash, Output, Sha256};
use tc_transact::{Gateway, IntoView, Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{label, path_label, NativeClass, PathLabel, PathSegment, TCPathBuf};

use super::{CacheBlock, State};

pub use class::*;
pub use instance::*;

mod class;
mod instance;
pub mod public;

const PREFIX: PathLabel = path_label(&["state", "object"]);

/// The type of a user-defined [`Object`].
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum ObjectType {
    Class,
    Instance,
}

impl tcgeneric::Class for ObjectType {}

impl NativeClass for ObjectType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() == 3 && &path[..2] == &PREFIX[..] {
            match path[2].as_str() {
                "class" => Some(Self::Class),
                "instance" => Some(Self::Instance),
                _ => None,
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        let suffix = match self {
            Self::Class => "class",
            Self::Instance => "instance",
        };

        TCPathBuf::from(PREFIX).append(label(suffix))
    }
}

impl fmt::Debug for ObjectType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Class => f.write_str("user-defined Class"),
            Self::Instance => f.write_str("user-defined Instance"),
        }
    }
}

/// A user-defined [`InstanceClass`] or [`InstanceExt`].
pub enum Object<Txn> {
    Class(InstanceClass),
    Instance(InstanceExt<Txn, State<Txn>>),
}

impl<Txn> Clone for Object<Txn> {
    fn clone(&self) -> Self {
        match self {
            Self::Class(class) => Self::Class(class.clone()),
            Self::Instance(instance) => Self::Instance(instance.clone()),
        }
    }
}

impl<Txn> tcgeneric::Instance for Object<Txn>
where
    Txn: Send + Sync,
{
    type Class = ObjectType;

    fn class(&self) -> ObjectType {
        match self {
            Self::Class(_) => ObjectType::Class,
            Self::Instance(_) => ObjectType::Instance,
        }
    }
}

#[async_trait]
impl<Txn> AsyncHash for Object<Txn>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
{
    async fn hash(&self, _txn_id: TxnId) -> TCResult<Output<Sha256>> {
        match self {
            Self::Class(class) => Ok(Hash::<Sha256>::hash(class)),
            Self::Instance(instance) => Err(bad_request!("cannot hash {:?}", instance)),
        }
    }
}

impl<Txn> From<InstanceClass> for Object<Txn> {
    fn from(class: InstanceClass) -> Self {
        Object::Class(class)
    }
}

impl<Txn> From<InstanceExt<Txn, State<Txn>>> for Object<Txn> {
    fn from(instance: InstanceExt<Txn, State<Txn>>) -> Self {
        Object::Instance(instance)
    }
}

impl<Txn> TryCastFrom<Object<Txn>> for Scalar {
    fn can_cast_from(object: &Object<Txn>) -> bool {
        match object {
            Object::Class(class) => Self::can_cast_from(class),
            Object::Instance(instance) => Self::can_cast_from(instance),
        }
    }

    fn opt_cast_from(object: Object<Txn>) -> Option<Self> {
        match object {
            Object::Class(class) => Self::opt_cast_from(class),
            Object::Instance(instance) => Self::opt_cast_from(instance),
        }
    }
}

impl<Txn> TryCastFrom<Object<Txn>> for Value {
    fn can_cast_from(object: &Object<Txn>) -> bool {
        match object {
            Object::Class(class) => Self::can_cast_from(class),
            Object::Instance(instance) => Self::can_cast_from(instance),
        }
    }

    fn opt_cast_from(object: Object<Txn>) -> Option<Self> {
        match object {
            Object::Class(class) => Self::opt_cast_from(class),
            Object::Instance(instance) => Self::opt_cast_from(instance),
        }
    }
}

#[async_trait]
impl<'en, Txn> IntoView<'en, CacheBlock> for Object<Txn>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
{
    type Txn = Txn;
    type View = ObjectView<'en>;

    async fn into_view(self, _txn: Txn) -> TCResult<ObjectView<'en>> {
        match self {
            Self::Class(class) => Ok(ObjectView::Class(class, PhantomData)),
            Self::Instance(_instance) => Err(not_implemented!("view of user-defined instance")),
        }
    }
}

#[async_trait]
impl<Txn> fmt::Debug for Object<Txn> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Class(ict) => fmt::Debug::fmt(ict, f),
            Self::Instance(ic) => fmt::Debug::fmt(ic, f),
        }
    }
}

/// A view of an [`Object`] at a specific transaction, used for serialization.
pub enum ObjectView<'en> {
    // the 'en lifetime is needed to compile when the collection feature flag is off
    Class(InstanceClass, PhantomData<&'en ()>),
}

impl<'en> en::IntoStream<'en> for ObjectView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        use destream::en::EncodeMap;

        let mut map = encoder.encode_map(Some(1))?;

        match self {
            Self::Class(class, _) => map.encode_entry(ObjectType::Class.path().to_string(), class),
        }?;

        map.end()
    }
}

/// A helper struct for Object deserialization
pub struct ObjectVisitor<Txn> {
    phantom: PhantomData<Txn>,
}

impl<Txn> ObjectVisitor<Txn>
where
    Txn: Send + Sync,
{
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }

    pub async fn visit_map_value<Err: de::Error>(
        self,
        class: ObjectType,
        state: State<Txn>,
    ) -> Result<Object<Txn>, Err> {
        match class {
            ObjectType::Class => {
                let class = state.try_cast_into(|s| {
                    de::Error::invalid_value(format!("{s:?}"), "a Class definition")
                })?;

                Ok(Object::Class(class))
            }
            ObjectType::Instance => Ok(Object::Instance(InstanceExt::new(
                state,
                InstanceClass::default(),
            ))),
        }
    }
}
