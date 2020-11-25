use std::collections::HashMap;
use std::fmt;
use std::iter::FromIterator;
use std::ops::{Deref, DerefMut};

use log::debug;
use serde::ser::{Serialize, Serializer};

use crate::class::{State, TCBoxTryFuture};
use crate::error;
use crate::request::Request;
use crate::transaction::Txn;

use super::{Id, PathSegment, Scalar, TCPath, Value, ValueInstance};

#[derive(Clone, Default, Eq, PartialEq)]
pub struct Object(HashMap<Id, Scalar>);

impl Object {
    pub fn get<'a>(
        &'a self,
        request: &'a Request,
        txn: &'a Txn,
        path: &'a [PathSegment],
        key: Value,
    ) -> TCBoxTryFuture<'a, State> {
        Box::pin(async move {
            debug!("Object::get {}: {}", TCPath::from(path), key);

            if path.is_empty() {
                return Ok(State::Scalar(Scalar::Object(self.clone())));
            }

            match self.0.get(&path[0]) {
                Some(scalar) => match scalar {
                    Scalar::Op(op_def) => {
                        op_def
                            .get(request, txn, key, Some(&self.clone().into()))
                            .await
                    }

                    Scalar::Value(value) => value
                        .get(&path[1..], key)
                        .map(Scalar::Value)
                        .map(State::Scalar),

                    other if path.len() == 1 => Ok(State::Scalar(other.clone())),
                    _ => Err(error::path_not_found(path)),
                },
                _ => Err(error::path_not_found(path)),
            }
        })
    }

    pub fn put<'a>(
        &'a self,
        _request: &'a Request,
        _txn: &'a Txn,
        _path: TCPath,
        _key: Value,
        _value: State,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move { Err(error::not_implemented("Object::put")) })
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

impl From<HashMap<Id, Scalar>> for Object {
    fn from(map: HashMap<Id, Scalar>) -> Object {
        Object(map)
    }
}

impl From<Object> for HashMap<Id, Scalar> {
    fn from(object: Object) -> HashMap<Id, Scalar> {
        object.0
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
