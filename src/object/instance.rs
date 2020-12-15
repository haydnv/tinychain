use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::ops::Deref;

use crate::class::{Instance, State};
use crate::handler::*;
use crate::scalar::{self, MethodType, PathSegment, Scalar};

use super::InstanceClass;

#[derive(Clone)]
pub struct InstanceExt<T: Instance> {
    parent: Box<T>,
    class: InstanceClass,
}

impl<T: Instance> InstanceExt<T> {
    pub fn new(parent: T, class: InstanceClass) -> InstanceExt<T> {
        InstanceExt {
            parent: Box::new(parent),
            class,
        }
    }

    pub fn into_state(self) -> InstanceExt<State>
    where
        State: From<T>,
    {
        let parent = Box::new((*self.parent).into());
        let class = self.class;
        InstanceExt { parent, class }
    }

    pub fn try_as<E, O: Instance + TryFrom<T, Error = E>>(self) -> Result<InstanceExt<O>, E> {
        let class = self.class;
        let parent = (*self.parent).try_into()?;

        Ok(InstanceExt {
            parent: Box::new(parent),
            class,
        })
    }
}

impl<T: Instance + Route> Route for InstanceExt<T>
where
    State: From<T>,
{
    fn route(&'_ self, method: MethodType, path: &[PathSegment]) -> Option<Box<dyn Handler + '_>> {
        let proto = self.class.proto().deref();
        match proto.get(&path[0]) {
            Some(scalar) => match scalar {
                Scalar::Op(op_def) if path.len() == 1 => {
                    Some(op_def.handler(Some(self.clone().into_state().into())))
                }
                scalar => scalar.route(method, path),
            },
            None => self.parent.route(method, path),
        }
    }
}

impl<T: Instance> Instance for InstanceExt<T> {
    type Class = InstanceClass;

    fn class(&self) -> Self::Class {
        self.class.clone()
    }
}

impl<T: Instance> From<T> for InstanceExt<T> {
    fn from(instance: T) -> InstanceExt<T> {
        let class = InstanceClass::from_class(instance.class());

        InstanceExt {
            parent: Box::new(instance),
            class,
        }
    }
}

impl From<scalar::Object> for InstanceExt<State> {
    fn from(scalar: scalar::Object) -> InstanceExt<State> {
        let class = InstanceClass::from_class(scalar.class());

        InstanceExt {
            parent: Box::new(State::Scalar(scalar.into())),
            class,
        }
    }
}

impl<T: Instance> fmt::Display for InstanceExt<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Object of type {}", self.class())
    }
}
