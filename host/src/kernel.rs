use error::*;
use generic::*;
use safecast::{Match, TryCastFrom};

use crate::scalar::*;
use crate::state::*;
use crate::txn::*;

const CAPTURE: Label = label("capture");

pub struct Kernel;

impl Kernel {
    pub async fn get(&self, _txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<State> {
        if let Some(class) = StateType::from_path(path) {
            let err = format!("Cannot cast into {} from {}", class, key);
            State::Scalar(Scalar::Value(key))
                .into_type(class)
                .ok_or_else(|| TCError::unsupported(err))
        } else {
            Err(TCError::not_found(TCPath::from(path)))
        }
    }

    pub async fn post(&self, txn: &Txn, path: &[PathSegment], data: State) -> TCResult<State> {
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
                        let executor = Executor::new(&txn, data);
                        executor.capture(capture).await
                    } else {
                        let executor = Executor::new(&txn, vec![(CAPTURE.into(), data)]);
                        executor.capture(CAPTURE.into()).await
                    }
                }
                "hypothetical" => Err(TCError::not_implemented("hypothetical queries")),
                other => Err(TCError::not_found(other)),
            },
            other => Err(TCError::not_found(other)),
        }
    }
}
