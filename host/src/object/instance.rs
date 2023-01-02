//! User-defined instance implementation.

use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::ops::Deref;

use async_trait::async_trait;
use destream::{en, EncodeMap};
use futures::future::TryFutureExt;
use futures::stream::{FuturesUnordered, TryStreamExt};
use log::debug;
use safecast::TryCastFrom;

use tc_error::*;
use tc_transact::IntoView;
use tc_value::{Link, Value};
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

#[async_trait]
impl<'en> IntoView<'en, Dir> for InstanceExt<State> {
    type Txn = Txn;
    type View = InstanceView<'en>;

    async fn into_view(self, txn: Txn) -> TCResult<InstanceView<'en>> {
        let mut members = if self.class.is_anonymous() {
            self.class.proto().clone().into_iter().collect()
        } else {
            Map::<State>::new()
        };

        for (id, state) in self.members {
            members.insert(id, state);
        }

        let mut into_view: FuturesUnordered<_> = members
            .into_iter()
            .map(|(id, member)| member.into_view(txn.clone()).map_ok(|view| (id, view)))
            .collect();

        let mut members = Map::new();
        while let Some((id, view)) = into_view.try_next().await? {
            members.insert(id, view);
        }

        Ok(InstanceView {
            class: self.class.link(),
            members,
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

impl<T: tcgeneric::Instance + fmt::Display> fmt::Display for InstanceExt<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} instance", tcgeneric::Instance::class(self))
    }
}

/// A view of an [`InstanceExt`] at a specific [`Txn`], used for serialization.
pub struct InstanceView<'en> {
    class: Link,
    members: Map<StateView<'en>>,
}

impl<'en> en::IntoStream<'en> for InstanceView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let mut map = encoder.encode_map(Some(1))?;
        map.encode_entry(self.class.to_string(), self.members)?;
        map.end()
    }
}
