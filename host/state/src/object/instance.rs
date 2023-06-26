//! User-defined instance implementation.

use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::ops::Deref;

use log::debug;
use safecast::TryCastFrom;

use tc_scalar::Scalar;
use tc_transact::public::ToState;
use tc_value::Value;
use tcgeneric::Map;

use crate::State;

use super::{InstanceClass, Object};

/// A user-defined instance, subclassing `T`.
#[derive(Clone)]
pub struct InstanceExt<T: tcgeneric::Instance> {
    parent: Box<T>,
    class: InstanceClass,
    members: Map<State>,
}

impl<T: tcgeneric::Instance> InstanceExt<T> {
    /// Construct a new instance of the given user-defined [`InstanceClass`].
    pub fn new(parent: T, class: InstanceClass) -> InstanceExt<T> {
        InstanceExt {
            parent: Box::new(parent),
            class,
            members: Map::default(),
        }
    }

    /// Construct a new instance of an anonymous class.
    pub fn anonymous(parent: T, class: InstanceClass, members: Map<State>) -> InstanceExt<T> {
        InstanceExt {
            parent: Box::new(parent),
            class,
            members,
        }
    }

    /// Borrow the members of this instance.
    pub fn members(&self) -> &Map<State> {
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
    ) -> Result<InstanceExt<O>, E> {
        let class = self.class;
        let parent = (*self.parent).try_into()?;

        Ok(InstanceExt {
            parent: Box::new(parent),
            class,
            members: self.members,
        })
    }
}

impl<T: tcgeneric::Instance> tcgeneric::Instance for InstanceExt<T> {
    type Class = InstanceClass;

    fn class(&self) -> Self::Class {
        self.class.clone()
    }
}

impl<T: tcgeneric::Instance> Deref for InstanceExt<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.parent
    }
}

impl<T: tcgeneric::Instance + fmt::Debug> TryCastFrom<InstanceExt<T>> for Scalar
where
    Scalar: TryCastFrom<T>,
{
    fn can_cast_from(instance: &InstanceExt<T>) -> bool {
        debug!("Scalar::can_cast_from {:?}?", instance);
        Self::can_cast_from(&(*instance).parent)
    }

    fn opt_cast_from(instance: InstanceExt<T>) -> Option<Self> {
        Self::opt_cast_from(*instance.parent)
    }
}

impl<T: tcgeneric::Instance + fmt::Debug> TryCastFrom<InstanceExt<T>> for Value
where
    Value: TryCastFrom<T>,
{
    fn can_cast_from(instance: &InstanceExt<T>) -> bool {
        debug!("Value::can_cast_from {:?}?", instance);
        Self::can_cast_from(&(*instance).parent)
    }

    fn opt_cast_from(instance: InstanceExt<T>) -> Option<Self> {
        Self::opt_cast_from(*instance.parent)
    }
}

impl<T: tcgeneric::Instance + ToState<State>> ToState<State> for InstanceExt<T> {
    fn to_state(&self) -> State {
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

impl<T: tcgeneric::Instance + fmt::Debug> fmt::Debug for InstanceExt<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?} instance", tcgeneric::Instance::class(self))
    }
}
