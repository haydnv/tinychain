use error::*;
use generic::*;
use safecast::{Match, TryCastFrom};

use crate::gateway::Request;
use crate::state::scalar::*;
use crate::state::*;
use crate::txn::Txn;

const CAPTURE: Label = label("capture");

pub struct Kernel;

impl Kernel {
    pub async fn get(
        &self,
        _request: Request,
        path: &[PathSegment],
        key: Value,
    ) -> TCResult<State> {
        if let Some(class) = StateType::from_path(path) {
            State::Scalar(Scalar::Value(key)).into_type(class)
        } else {
            Err(TCError::not_found(TCPath::from(path)))
        }
    }

    pub async fn post(
        &self,
        request: Request,
        path: &[PathSegment],
        data: State,
    ) -> TCResult<State> {
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
                        let mut txn = Txn::new(data, request.txn_id);
                        txn.execute(capture).await
                    } else {
                        let mut txn = Txn::new(vec![(CAPTURE.into(), data)], request.txn_id);
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
