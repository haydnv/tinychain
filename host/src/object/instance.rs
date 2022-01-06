//! User-defined instance implementation.

use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::ops::Deref;

use async_hash::Hash;
use async_trait::async_trait;
use destream::{en, EncodeMap};
use log::debug;
use safecast::TryCastFrom;
use sha2::digest::{Digest, Output};
use sha2::Sha256;

use tc_error::*;
use tc_transact::IntoView;
use tc_value::Value;
use tcgeneric::Map;

use crate::fs::Dir;
use crate::scalar::Scalar;
use crate::state::{State, StateView, ToState};
use crate::txn::Txn;

use super::{InstanceClass, Object};

/// A user-defined instance, subclassing `T`.
#[derive(Clone)]
pub struct InstanceExt<T: tcgeneric::Instance> {
    parent: Box<T>,
    class: InstanceClass,
}

impl<T: tcgeneric::Instance> InstanceExt<T> {
    /// Construct a new instance of the given user-defined [`InstanceClass`].
    pub fn new(parent: T, class: InstanceClass) -> InstanceExt<T> {
        InstanceExt {
            parent: Box::new(parent),
            class,
        }
    }

    /// Borrow the parent of this instance.
    pub fn parent(&self) -> &T {
        &self.parent
    }

    /// Borrow the class prototype of this instance.
    pub fn proto(&self) -> &Map<Scalar> {
        self.class.proto()
    }

    /// Convert the native type of this instance, if possible.
    pub fn try_into<E, O: tcgeneric::Instance + TryFrom<T, Error = E>>(
        self,
    ) -> Result<InstanceExt<O>, E> {
        let class = self.class;
        let parent = (*self.parent).try_into()?;

        Ok(InstanceExt {
            parent: Box::new(parent),
            class,
        })
    }
}

impl<T: tcgeneric::Instance> tcgeneric::Instance for InstanceExt<T> {
    type Class = InstanceClass;

    fn class(&self) -> Self::Class {
        self.class.clone()
    }
}

impl<'en, T: tcgeneric::Instance + en::IntoStream<'en> + 'en> en::IntoStream<'en>
    for InstanceExt<T>
{
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let mut map = encoder.encode_map(Some(1))?;
        map.encode_entry(self.class.extends().to_string(), self.parent)?;
        map.end()
    }
}

impl<T: tcgeneric::Instance> Deref for InstanceExt<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.parent
    }
}

impl InstanceExt<State> {
    pub async fn hash(self, txn: Txn) -> TCResult<Output<Sha256>> {
        let parent = self.parent.hash(txn).await?;

        let mut hasher = Sha256::default();
        hasher.update(Hash::<Sha256>::hash(self.class));
        hasher.update(&parent);
        Ok(hasher.finalize())
    }
}

#[async_trait]
impl<'en> IntoView<'en, Dir> for InstanceExt<State> {
    type Txn = Txn;
    type View = InstanceView<'en>;

    async fn into_view(self, txn: Txn) -> TCResult<InstanceView<'en>> {
        Ok(InstanceView {
            class: self.class,
            parent: self.parent.into_view(txn).await?,
        })
    }
}

impl<T: tcgeneric::Instance + fmt::Display> TryCastFrom<InstanceExt<T>> for Scalar
where
    Scalar: TryCastFrom<T>,
{
    fn can_cast_from(instance: &InstanceExt<T>) -> bool {
        debug!("Scalar::can_cast_from {}?", instance);

        Self::can_cast_from(&(*instance).parent)
    }

    fn opt_cast_from(instance: InstanceExt<T>) -> Option<Self> {
        Self::opt_cast_from(*instance.parent)
    }
}

impl<T: tcgeneric::Instance + fmt::Display> TryCastFrom<InstanceExt<T>> for Value
where
    Value: TryCastFrom<T>,
{
    fn can_cast_from(instance: &InstanceExt<T>) -> bool {
        debug!("Value::can_cast_from {}?", instance);

        Self::can_cast_from(&(*instance).parent)
    }

    fn opt_cast_from(instance: InstanceExt<T>) -> Option<Self> {
        Self::opt_cast_from(*instance.parent)
    }
}

impl<T: tcgeneric::Instance + ToState> ToState for InstanceExt<T> {
    fn to_state(&self) -> State {
        let parent = Box::new(self.parent.to_state());
        let class = self.class.clone();
        let instance = InstanceExt { parent, class };
        State::Object(Object::Instance(instance))
    }
}

impl<T: tcgeneric::Instance + fmt::Debug> fmt::Debug for InstanceExt<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{:?} instance: {:?}",
            tcgeneric::Instance::class(self),
            self.parent
        )
    }
}

impl<T: tcgeneric::Instance + fmt::Display> fmt::Display for InstanceExt<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} instance: {}",
            tcgeneric::Instance::class(self),
            self.parent
        )
    }
}

/// A view of an [`InstanceExt`] at a specific [`Txn`], used for serialization.
pub struct InstanceView<'en> {
    class: InstanceClass,
    parent: StateView<'en>,
}

impl<'en> en::IntoStream<'en> for InstanceView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let mut map = encoder.encode_map(Some(1))?;
        map.encode_entry(self.class.extends().to_string(), self.parent)?;
        map.end()
    }
}
