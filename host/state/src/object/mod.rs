//! User-defined public-orientation features.

use std::fmt;
use std::marker::PhantomData;

use async_hash::{Output, Sha256};
use async_trait::async_trait;
use destream::{de, en};
use safecast::{AsType, TryCastFrom, TryCastInto};
use tc_chain::ChainBlock;
use tc_collection::{BTreeNode, DenseCacheFile, TensorNode};

use tc_error::*;
use tc_scalar::Scalar;
use tc_transact::{fs, AsyncHash, IntoView, Gateway, Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{label, path_label, NativeClass, PathLabel, PathSegment, TCPathBuf};

use super::State;

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
pub enum Object<Txn, FE> {
    Class(InstanceClass),
    Instance(InstanceExt<Txn, FE, State<Txn, FE>>),
}

impl<Txn, FE> Clone for Object<Txn, FE> {
    fn clone(&self) -> Self {
        match self {
            Self::Class(class) => Self::Class(class.clone()),
            Self::Instance(instance) => Self::Instance(instance.clone()),
        }
    }
}

impl<Txn, FE> tcgeneric::Instance for Object<Txn, FE>
where
    Txn: Send + Sync,
    FE: Send + Sync,
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
impl<Txn, FE> AsyncHash for Object<Txn, FE>
where
    Txn: Transaction<FE> + Gateway<State<Txn, FE>>,
    FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<TensorNode>
        + AsType<ChainBlock>
        + for<'a> fs::FileSave<'a>
        + Clone,
{
    async fn hash(self, _txn_id: TxnId) -> TCResult<Output<Sha256>> {
        match self {
            Self::Class(class) => Ok(async_hash::Hash::<Sha256>::hash(class)),
            Self::Instance(instance) => Err(bad_request!("cannot hash {:?}", instance)),
        }
    }
}

impl<Txn, FE> From<InstanceClass> for Object<Txn, FE> {
    fn from(class: InstanceClass) -> Self {
        Object::Class(class)
    }
}

impl<Txn, FE> From<InstanceExt<Txn, FE, State<Txn, FE>>> for Object<Txn, FE> {
    fn from(instance: InstanceExt<Txn, FE, State<Txn, FE>>) -> Self {
        Object::Instance(instance)
    }
}

impl<Txn, FE> TryCastFrom<Object<Txn, FE>> for Scalar {
    fn can_cast_from(object: &Object<Txn, FE>) -> bool {
        match object {
            Object::Class(class) => Self::can_cast_from(class),
            Object::Instance(instance) => Self::can_cast_from(instance),
        }
    }

    fn opt_cast_from(object: Object<Txn, FE>) -> Option<Self> {
        match object {
            Object::Class(class) => Self::opt_cast_from(class),
            Object::Instance(instance) => Self::opt_cast_from(instance),
        }
    }
}

impl<Txn, FE> TryCastFrom<Object<Txn, FE>> for Value {
    fn can_cast_from(object: &Object<Txn, FE>) -> bool {
        match object {
            Object::Class(class) => Self::can_cast_from(class),
            Object::Instance(instance) => Self::can_cast_from(instance),
        }
    }

    fn opt_cast_from(object: Object<Txn, FE>) -> Option<Self> {
        match object {
            Object::Class(class) => Self::opt_cast_from(class),
            Object::Instance(instance) => Self::opt_cast_from(instance),
        }
    }
}

#[async_trait]
impl<'en, Txn, FE> IntoView<'en, FE> for Object<Txn, FE>
where
    Txn: Transaction<FE> + Gateway<State<Txn, FE>>,
    FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'a> fs::FileSave<'a>
        + Clone,
{
    type Txn = Txn;
    type View = ObjectView;

    async fn into_view(self, _txn: Txn) -> TCResult<ObjectView> {
        match self {
            Self::Class(class) => Ok(ObjectView::Class(class)),
            Self::Instance(_instance) => Err(not_implemented!("view of user-defined instance")),
        }
    }
}

#[async_trait]
impl<Txn, FE> fmt::Debug for Object<Txn, FE> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Class(ict) => fmt::Debug::fmt(ict, f),
            Self::Instance(ic) => fmt::Debug::fmt(ic, f),
        }
    }
}

/// A view of an [`Object`] at a specific [`Txn`], used for serialization.
pub enum ObjectView {
    Class(InstanceClass),
}

impl<'en> en::IntoStream<'en> for ObjectView {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        use destream::en::EncodeMap;

        let mut map = encoder.encode_map(Some(1))?;

        match self {
            Self::Class(class) => map.encode_entry(ObjectType::Class.path().to_string(), class),
        }?;

        map.end()
    }
}

/// A helper struct for Object deserialization
pub struct ObjectVisitor<Txn, FE> {
    phantom: PhantomData<(Txn, FE)>,
}

impl<Txn, FE> ObjectVisitor<Txn, FE>
where
    Txn: Send + Sync,
    FE: Send + Sync,
{
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }

    pub async fn visit_map_value<Err: de::Error>(
        self,
        class: ObjectType,
        state: State<Txn, FE>,
    ) -> Result<Object<Txn, FE>, Err> {
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
