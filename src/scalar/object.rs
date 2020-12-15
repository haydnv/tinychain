use std::collections::HashMap;
use std::fmt;
use std::iter::FromIterator;
use std::ops::{Deref, DerefMut};

use serde::ser::{Serialize, Serializer};

use crate::class::{Class, Instance, NativeClass, State, TCResult, TCType};
use crate::error;
use crate::handler::*;

use super::{
    label, Id, Link, PathSegment, Scalar, ScalarInstance, ScalarType, TCPathBuf, TryCastFrom,
    TryCastInto,
};
use crate::scalar::MethodType;

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct ObjectType;

impl Class for ObjectType {
    type Instance = Object;
}

impl NativeClass for ObjectType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        if path == Self::prefix().as_slice() {
            Ok(Self)
        } else {
            Err(error::path_not_found(path))
        }
    }

    fn prefix() -> TCPathBuf {
        ScalarType::prefix().append(label("object"))
    }
}

impl From<ObjectType> for Link {
    fn from(_ot: ObjectType) -> Link {
        ObjectType::prefix().into()
    }
}

impl From<ObjectType> for TCType {
    fn from(_: ObjectType) -> TCType {
        ScalarType::Object.into()
    }
}

impl fmt::Display for ObjectType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "generic Object type")
    }
}

#[derive(Clone, Default, Eq, PartialEq)]
pub struct Object(HashMap<Id, Scalar>);

impl Instance for Object {
    type Class = ObjectType;

    fn class(&self) -> Self::Class {
        ObjectType
    }
}

impl Route for Object {
    fn route(&'_ self, method: MethodType, path: &[PathSegment]) -> Option<Box<dyn Handler + '_>> {
        if path.is_empty() {
            return None;
        }

        if let Some(scalar) = self.0.get(&path[0]) {
            match scalar {
                Scalar::Op(op_def) if path.len() == 1 => {
                    Some(op_def.handler(Some(self.clone().into())))
                }
                scalar => scalar.route(method, &path[1..]),
            }
        } else {
            None
        }
    }
}

impl Deref for Object {
    type Target = HashMap<Id, Scalar>;

    fn deref(&'_ self) -> &'_ HashMap<Id, Scalar> {
        &self.0
    }
}

impl DerefMut for Object {
    fn deref_mut(&'_ mut self) -> &'_ mut HashMap<Id, Scalar> {
        &mut self.0
    }
}

impl<T: Into<Scalar>> FromIterator<(Id, T)> for Object {
    fn from_iter<I: IntoIterator<Item = (Id, T)>>(iter: I) -> Self {
        let mut object = HashMap::new();

        for (id, attr) in iter {
            let scalar = attr.into();
            object.insert(id, scalar);
        }

        Object(object)
    }
}

impl From<HashMap<Id, Scalar>> for Object {
    fn from(map: HashMap<Id, Scalar>) -> Object {
        Object(map)
    }
}

impl TryCastFrom<Scalar> for Object {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Object(_) => true,
            other => other.matches::<Vec<(Id, Scalar)>>(),
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Object> {
        match scalar {
            Scalar::Object(object) => Some(object),
            other if other.matches::<Vec<(Id, Scalar)>>() => {
                let data: Vec<(Id, Scalar)> = other.opt_cast_into().unwrap();
                Some(Object::from_iter(data))
            }
            _ => None,
        }
    }
}

impl TryCastFrom<State> for Object {
    fn can_cast_from(state: &State) -> bool {
        if let State::Scalar(scalar) = state {
            Object::can_cast_from(scalar)
        } else {
            false
        }
    }

    fn opt_cast_from(state: State) -> Option<Object> {
        if let State::Scalar(scalar) = state {
            Object::opt_cast_from(scalar)
        } else {
            None
        }
    }
}

impl From<Object> for HashMap<Id, Scalar> {
    fn from(object: Object) -> HashMap<Id, Scalar> {
        object.0
    }
}

impl From<Object> for State {
    fn from(object: Object) -> State {
        State::Scalar(Scalar::Object(object))
    }
}

impl IntoIterator for Object {
    type Item = (Id, Scalar);
    type IntoIter = std::collections::hash_map::IntoIter<Id, Scalar>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Serialize for Object {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(s)
    }
}

impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{{{}}}",
            self.0
                .iter()
                .map(|(k, v)| format!("{}: {}", k, v))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}
