use error::*;
use generic::PathSegment;

use crate::state::State;
use crate::txn::TxnId;

pub struct Kernel;

impl Kernel {
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
