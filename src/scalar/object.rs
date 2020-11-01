use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use serde::ser::{Serialize, Serializer};

use crate::auth::Auth;
use crate::class::{State, TCBoxTryFuture};
use crate::error;
use crate::transaction::Txn;

use super::{Op, Scalar, TCPath, Value, ValueId, ValueInstance};

#[derive(Clone, Default, Eq, PartialEq)]
pub struct Object(HashMap<ValueId, Scalar>);

impl Object {
    pub fn data(&'_ self) -> &'_ HashMap<ValueId, Scalar> {
        &self.0
    }

    pub fn get<'a>(
        &'a self,
        txn: Arc<Txn>,
        path: TCPath,
        key: Value,
        auth: Auth,
    ) -> TCBoxTryFuture<'a, State> {
        Box::pin(async move {
            println!("Object::get {}: {}", path, key);

            if path.is_empty() {
                return Ok(State::Scalar(Scalar::Object(self.clone())));
            }

            println!(
                "data: [{}]",
                self.0
                    .keys()
                    .map(|id| id.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            );

            match self.0.get(&path[0]) {
                Some(scalar) => match scalar {
                    Scalar::Op(op) => match &**op {
                        Op::Def(op_def) => {
                            op_def.get(txn, key, auth, Some(&self.clone().into())).await
                        }
                        other => Err(error::not_implemented(other)),
                    },

                    Scalar::Value(value) => value
                        .get(path.slice_from(1), key)
                        .map(Scalar::Value)
                        .map(State::Scalar),

                    other if path.len() == 1 => Ok(State::Scalar(other.clone())),
                    _ => Err(error::not_found(path)),
                },
                _ => Err(error::not_found(path)),
            }
        })
    }

    pub fn put<'a>(
        &'a self,
        _txn: Arc<Txn>,
        _path: TCPath,
        _key: Value,
        _value: State,
        _auth: Auth,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move { Err(error::not_implemented("Object::put")) })
    }
}

impl IntoIterator for Object {
    type Item = (ValueId, Scalar);
    type IntoIter = std::collections::hash_map::IntoIter<ValueId, Scalar>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Serialize for Object {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(s)
    }
}

impl From<HashMap<ValueId, Scalar>> for Object {
    fn from(map: HashMap<ValueId, Scalar>) -> Object {
        Object(map)
    }
}

impl From<Object> for HashMap<ValueId, Scalar> {
    fn from(object: Object) -> HashMap<ValueId, Scalar> {
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
