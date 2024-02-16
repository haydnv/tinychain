//! User-defined instance implementation.

use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::ops::Deref;

use log::debug;
use safecast::TryCastFrom;

use tc_scalar::Scalar;
use tc_transact::public::ToState;
use tc_transact::{Gateway, Transaction};
use tc_value::Value;
use tcgeneric::Map;

use crate::{CacheBlock, State};

use super::{InstanceClass, Object};

/// A user-defined instance, subclassing `T`.
pub struct InstanceExt<Txn, T> {
    parent: Box<T>,
    class: InstanceClass,
    members: Map<State<Txn>>,
}

impl<Txn, T> Clone for InstanceExt<Txn, T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            parent: self.parent.clone(),
            class: self.class.clone(),
            members: self.members.clone(),
        }
    }
}

impl<Txn, T> InstanceExt<Txn, T>
where
    T: tcgeneric::Instance,
{
    /// Construct a new instance of the given user-defined [`InstanceClass`].
    pub fn new(parent: T, class: InstanceClass) -> Self {
        InstanceExt {
            parent: Box::new(parent),
            class,
            members: Map::default(),
        }
    }

    /// Construct a new instance of an anonymous class.
    pub fn anonymous(parent: T, class: InstanceClass, members: Map<State<Txn>>) -> Self {
        InstanceExt {
            parent: Box::new(parent),
            class,
            members,
        }
    }

    /// Borrow the members of this instance.
    pub fn members(&self) -> &Map<State<Txn>> {
        &self.members
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
    ) -> Result<InstanceExt<Txn, O>, E> {
        let class = self.class;
        let parent = (*self.parent).try_into()?;

        Ok(InstanceExt {
            parent: Box::new(parent),
            class,
            members: self.members,
        })
    }
}

impl<Txn, T> tcgeneric::Instance for InstanceExt<Txn, T>
where
    Txn: Send + Sync,
    T: tcgeneric::Instance,
{
    type Class = InstanceClass;

    fn class(&self) -> Self::Class {
        self.class.clone()
    }
}

impl<Txn, T> Deref for InstanceExt<Txn, T>
where
    T: tcgeneric::Instance,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.parent
    }
}

impl<Txn, T> TryCastFrom<InstanceExt<Txn, T>> for Scalar
where
    T: fmt::Debug,
    Scalar: TryCastFrom<T>,
{
    fn can_cast_from(instance: &InstanceExt<Txn, T>) -> bool {
        debug!("Scalar::can_cast_from {:?}?", instance);
        Self::can_cast_from(&(*instance).parent)
    }

    fn opt_cast_from(instance: InstanceExt<Txn, T>) -> Option<Self> {
        Self::opt_cast_from(*instance.parent)
    }
}

impl<Txn, T> TryCastFrom<InstanceExt<Txn, T>> for Value
where
    T: fmt::Debug,
    Value: TryCastFrom<T>,
{
    fn can_cast_from(instance: &InstanceExt<Txn, T>) -> bool {
        debug!("Value::can_cast_from {:?}?", instance);
        Self::can_cast_from(&(*instance).parent)
    }

    fn opt_cast_from(instance: InstanceExt<Txn, T>) -> Option<Self> {
        Self::opt_cast_from(*instance.parent)
    }
}

impl<Txn, T> ToState<State<Txn>> for InstanceExt<Txn, T>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
    T: tcgeneric::Instance + ToState<State<Txn>>,
{
    fn to_state(&self) -> State<Txn> {
        let parent = Box::new(self.parent.to_state());
        let class = self.class.clone();
        let members = self.members.clone();
        let instance = InstanceExt {
            parent,
            class,
            members,
        };

        State::Object(Object::Instance(instance))
    }
}

impl<Txn, T> fmt::Debug for InstanceExt<Txn, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "instance of {}", std::any::type_name::<T>())
    }
}
