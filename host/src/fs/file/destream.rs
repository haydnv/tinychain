use std::convert::TryFrom;

use async_trait::async_trait;
use destream::*;
use futures::TryFutureExt;
use log::debug;

use tc_error::*;
use tc_transact::fs;

use crate::chain::ChainBlock;
use crate::fs::cache::*;
use crate::txn::TxnId;

use super::File;

struct FileVisitor<B> {
    txn_id: TxnId,
    file: File<B>,
}

#[async_trait]
impl<B: fs::BlockData + FromStream<Context = ()> + 'static> Visitor for FileVisitor<B>
where
    CacheBlock: From<CacheLock<B>>,
    CacheLock<B>: TryFrom<CacheBlock, Error = TCError>,
{
    type Value = File<B>;

    fn expecting() -> &'static str {
        "a File"
    }

    async fn visit_seq<A: SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut i = 0u64;
        while let Some(block) = seq
            .next_element(())
            .map_err(|e| de::Error::custom(format!("invalid block: {}", e)))
            .await?
        {
            debug!("decoded file block {}", i);
            fs::File::create_block(&self.file, self.txn_id, i.into(), block)
                .map_err(de::Error::custom)
                .await?;

            i += 1;
            debug!("checking whether to decode file block {}...", i);
        }

        Ok(self.file)
    }
}

#[async_trait]
impl FromStream for File<ChainBlock> {
    type Context = (TxnId, File<ChainBlock>);

    async fn from_stream<D: Decoder>(
        cxt: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        let visitor = FileVisitor {
            txn_id: cxt.0,
            file: cxt.1,
        };

        decoder.decode_seq(visitor).await
    }
}
