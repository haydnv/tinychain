use async_trait::async_trait;
use destream::de;

use error::*;
use generic::{label, Instance, Label};
use transact::fs::File;
use transact::lock::TxnLock;
use transact::TxnId;

use crate::fs;
use crate::scalar::OpRef;
use crate::state::State;
use crate::txn::Txn;

use super::{ChainBlock, ChainInstance, Subject};

const BLOCK_ID: Label = label("0");

#[derive(Clone)]
pub struct SyncChain {
    subject: Subject,
    file: fs::File<ChainBlock>,
}

impl SyncChain {
    pub async fn load(subject: State, file: fs::File<ChainBlock>) -> TCResult<Self> {
        // TODO: validate file

        let subject_class = subject.class();
        let subject = if let State::Scalar(subject) = subject {
            Subject::Scalar(TxnLock::new("sync chain subject", subject.into()))
        } else {
            return Err(TCError::bad_request(
                "Chain does not support the given subject",
                subject_class,
            ));
        };

        Ok(Self { subject, file })
    }
}

#[async_trait]
impl ChainInstance for SyncChain {
    async fn append(&self, txn_id: &TxnId, op_ref: OpRef) -> TCResult<()> {
        let block_id = BLOCK_ID.into();
        let block = self.file.get_block(txn_id, &block_id).await?;
        let mut block = block.upgrade().await?;
        block.append(op_ref);
        Ok(())
    }
}

#[async_trait]
impl de::FromStream for SyncChain {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(_txn: Txn, _decoder: &mut D) -> Result<Self, D::Error> {
        unimplemented!()
    }
}
