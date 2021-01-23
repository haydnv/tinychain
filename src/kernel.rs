use error::*;
use generic::*;
use number_general::{Number, NumberInstance};
use safecast::{Match, TryCastFrom, TryCastInto};
use scalar::*;

use crate::state::{State, StateType};
use crate::txn::*;

const CAPTURE: Label = label("capture");

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

    pub async fn post(&self, txn_id: TxnId, path: &[PathSegment], data: State) -> TCResult<State> {
        if path.is_empty() {
            return Err(TCError::method_not_allowed(TCPath::from(path)));
        }

        match path[0].as_str() {
            "transact" if path.len() == 1 => Err(TCError::method_not_allowed(path[0].as_str())),
            "transact" if path.len() == 2 => match path[1].as_str() {
                "execute" => {
                    if data.matches::<Tuple<(Id, State)>>() {
                        let data = Tuple::<(Id, State)>::try_cast_from(data, |s| {
                            TCError::bad_request(
                                "A transaction is a list of (Id, State) tuples, not",
                                s,
                            )
                        })?;
                        if data.is_empty() {
                            return Ok(State::Tuple(Tuple::default()));
                        }

                        let capture = data.last().unwrap().0.clone();
                        let mut txn = Txn::new(data, txn_id);
                        txn.execute(capture).await
                    } else {
                        let mut txn = Txn::new(vec![(CAPTURE.into(), data)], txn_id);
                        txn.execute(CAPTURE.into()).await
                    }
                }
                "hypothetical" => Err(TCError::not_implemented("hypothetical queries")),
                other => Err(TCError::not_found(other)),
            },
            other => Err(TCError::not_found(other)),
        }
    }
}
