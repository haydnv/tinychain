use std::convert::{TryFrom, TryInto};
use std::fmt;

use async_trait::async_trait;
use destream::de;

use crate::state::State;

use super::{InstanceClass, Object};

#[derive(Clone)]
pub struct InstanceExt<T: generic::Instance> {
    parent: Box<T>,
    class: InstanceClass,
}

impl<T: generic::Instance> InstanceExt<T> {
    pub fn new(parent: T, class: InstanceClass) -> InstanceExt<T> {
        InstanceExt {
            parent: Box::new(parent),
            class,
        }
    }

    pub fn try_into<E, O: generic::Instance + TryFrom<T, Error = E>>(
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

impl<T: generic::Instance> generic::Instance for InstanceExt<T> {
    type Class = InstanceClass;

    fn class(&self) -> Self::Class {
        self.class.clone()
    }
}

impl<T: generic::Instance> From<InstanceExt<T>> for State
where
    State: From<T>,
{
    fn from(instance: InstanceExt<T>) -> State {
        let parent = Box::new((*instance.parent).into());
        let class = instance.class;
        let instance = InstanceExt { parent, class };
        State::Object(Object::Instance(instance))
    }
}

impl<T: generic::Instance> fmt::Display for InstanceExt<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} Object", generic::Instance::class(self))
    }
}
