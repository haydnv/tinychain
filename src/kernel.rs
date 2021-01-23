use error::*;
use generic::{NativeClass, PathSegment, TCPath};
use number_general::{Number, NumberInstance};
use safecast::{TryCastFrom, TryCastInto};
use scalar::*;

use crate::state::{State, StateType};
use crate::txn::TxnId;

pub struct Kernel;

impl Kernel {
    pub async fn get(&self, _txn_id: TxnId, path: &[PathSegment], key: Value) -> TCResult<State> {
        use ValueType as VT;

        if let Some(class) = StateType::from_path(path) {
            match class {
                StateType::Scalar(class) => match class {
                    ScalarType::Value(class) => match class {
                        VT::Link => {
                            let l = Link::try_cast_from(key, |v| {
                                TCError::bad_request("Cannot cast into Link from {}", v)
                            })?;
                            Ok(Value::Link(l).into())
                        }
                        VT::None => Ok(Value::None.into()),
                        VT::Number(nt) => {
                            let n: Number = key.try_cast_into(|v| {
                                TCError::bad_request("Cannot cast into Number from {}", v)
                            })?;
                            let n = n.into_type(nt);
                            Ok(Value::Number(n).into())
                        }
                        VT::String => Ok(Value::String(key.to_string()).into()),
                        VT::Tuple => match key {
                            Value::Tuple(t) => Ok(Value::Tuple(t).into()),
                            other => Ok(Value::Tuple(vec![other].into()).into()),
                        },
                        VT::Value => Ok(key.into()),
                    },
                    other => Err(TCError::not_implemented(other)),
                },
                other => Err(TCError::not_implemented(other)),
            }
        } else {
            Err(TCError::not_found(TCPath::from(path)))
        }
    }

    pub async fn post<T>(
        &self,
        _txn_id: TxnId,
        _path: &[PathSegment],
        _data: T,
    ) -> TCResult<State> {
        Ok(State::Scalar(scalar::Scalar::Value(scalar::Value::String(
            "Hello, world!".into(),
        ))))
    }
}
