use std::fmt;

use error::*;
use generic::*;
use number_general::{Number, NumberInstance};
use safecast::{Match, TryCastFrom, TryCastInto};

use crate::scalar::*;
use crate::state::*;
use crate::txn::*;

const CAPTURE: Label = label("capture");

pub struct Kernel;

impl Kernel {
    pub async fn get(&self, _txn_id: TxnId, path: &[PathSegment], key: Value) -> TCResult<State> {
        use OpDefType as ODT;
        use OpRefType as ORT;
        use RefType as RT;
        use ScalarType as ST;
        use ValueType as VT;

        if let Some(class) = StateType::from_path(path) {
            match class {
                StateType::Scalar(class) => match class {
                    ST::Map => Err(cast_err(ST::Map, &key)),
                    ST::Op(ot) => match ot {
                        ODT::Get => {
                            let op = key.try_cast_into(try_cast_err(ODT::Get))?;
                            Ok(Scalar::Op(OpDef::Get(op)).into())
                        }
                        ODT::Put => {
                            let op = key.try_cast_into(try_cast_err(ODT::Put))?;
                            Ok(Scalar::Op(OpDef::Put(op)).into())
                        }
                        ODT::Post => {
                            let op = key.try_cast_into(try_cast_err(ODT::Post))?;
                            Ok(Scalar::Op(OpDef::Post(op)).into())
                        }
                        ODT::Delete => {
                            let op = key.try_cast_into(try_cast_err(ODT::Delete))?;
                            Ok(Scalar::Op(OpDef::Delete(op)).into())
                        }
                    },
                    ScalarType::Value(class) => match class {
                        VT::Link => {
                            let l = Link::try_cast_from(key, try_cast_err(VT::Link))?;
                            Ok(Value::Link(l).into())
                        }
                        VT::None => Ok(Value::None.into()),
                        VT::Number(nt) => {
                            let n: Number = key.try_cast_into(try_cast_err(nt))?;
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
                    ST::Tuple => {
                        let tuple = Tuple::<Scalar>::try_cast_from(key, try_cast_err(ST::Tuple))?;
                        Ok(Scalar::Tuple(tuple).into())
                    }
                },
                StateType::Ref(rt) => match rt {
                    RT::Id => {
                        let id_ref = IdRef::try_cast_from(key, try_cast_err(RT::Id))?;
                        Ok(State::from(id_ref))
                    }
                    RT::Op(ort) => match ort {
                        ORT::Get => {
                            let get_ref = GetRef::try_cast_from(key, try_cast_err(ORT::Get))?;
                            Ok(State::from(OpRef::Get(get_ref)))
                        }
                        ORT::Put => {
                            let put_ref = PutRef::try_cast_from(key, try_cast_err(ORT::Put))?;
                            Ok(State::from(OpRef::Put(put_ref)))
                        }
                        ORT::Post => {
                            let post_ref = PostRef::try_cast_from(key, try_cast_err(ORT::Put))?;
                            Ok(State::from(OpRef::Post(post_ref)))
                        }
                        ORT::Delete => {
                            let delete_ref = DeleteRef::try_cast_from(key, try_cast_err(ORT::Put))?;
                            Ok(State::from(OpRef::Delete(delete_ref)))
                        }
                    },
                },
                other => Err(TCError::bad_request(
                    format!("Cannot cast into {} from", other),
                    key,
                )),
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

fn cast_err<T: fmt::Display>(to: T, from: &Value) -> TCError {
    TCError::bad_request(format!("Cannot cast into {} from", to), from)
}

fn try_cast_err<T: fmt::Display>(to: T) -> impl FnOnce(&Value) -> TCError {
    move |v| cast_err(to, v)
}
