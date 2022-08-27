use async_trait::async_trait;
use futures::{TryFutureExt, TryStreamExt};

use tc_error::{TCError, TCResult};
use tc_transact::fs::{DirLock, FileLock};
use tc_transact::{Transaction, TxnId};
use tc_value::ValueCollator;
use tcgeneric::{Instance, TCBoxTryStream};

use super::{
    validate_range, BTree, BTreeFile, BTreeInstance, BTreeType, Key, Node, Range, RowSchema,
};

/// A slice of a [`BTree`]
#[derive(Clone)]
pub struct BTreeSlice<F, D, T> {
    source: BTreeFile<F, D, T>,
    range: Range,
    reverse: bool,
}

impl<F: FileLock<Block = Node>, D: DirLock<File = F>, T: Transaction<D>> BTreeSlice<F, D, T> {
    pub fn new(
        source: BTree<F, D, T>,
        range: Range,
        reverse: bool,
    ) -> TCResult<BTreeSlice<F, D, T>> {
        let range = validate_range(range, source.schema())?;

        match source {
            BTree::File(tree) => Ok(Self {
                source: tree,
                range,
                reverse,
            }),

            BTree::Slice(view) => {
                let source = view.source.clone();
                let reverse = view.reverse ^ reverse;

                if range == Range::default() {
                    Ok(Self {
                        source,
                        range: view.range,
                        reverse,
                    })
                } else if view.range.contains(&range, view.source.collator()) {
                    Ok(Self {
                        source,
                        range,
                        reverse,
                    })
                } else {
                    Err(TCError::unsupported(
                        "BTreeSlice does not contain requested range",
                    ))
                }
            }
        }
    }
}

impl<F, D, T> Instance for BTreeSlice<F, D, T>
where
    BTreeFile<F, D, T>: Send + Sync,
{
    type Class = BTreeType;

    fn class(&self) -> Self::Class {
        BTreeType::Slice
    }
}

#[async_trait]
impl<F: FileLock<Block = Node>, D: DirLock<File = F>, T: Transaction<D>> BTreeInstance
    for BTreeSlice<F, D, T>
where
    BTreeFile<F, D, T>: Clone + 'static,
{
    type Slice = Self;

    fn collator(&'_ self) -> &'_ ValueCollator {
        self.source.collator()
    }

    fn schema(&'_ self) -> &'_ RowSchema {
        self.source.schema()
    }

    fn slice(self, range: Range, reverse: bool) -> TCResult<Self::Slice> {
        let range = validate_range(range, self.schema())?;

        if self.range.contains(&range, self.collator()) {
            Self::new(BTree::Slice(self), range, reverse)
        } else {
            Err(TCError::unsupported(
                "BTreeSlice does not contain the requested range",
            ))
        }
    }

    async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        let mut rows = self
            .source
            .clone()
            .rows_in_range(txn_id, self.range.clone(), self.reverse)
            .await?;

        rows.try_next().map_ok(|row| row.is_none()).await
    }

    async fn keys<'a>(self, txn_id: TxnId) -> TCResult<TCBoxTryStream<'a, Key>> {
        self.source
            .rows_in_range(txn_id, self.range, self.reverse)
            .await
    }

    fn validate_key(&self, key: Key) -> TCResult<Key> {
        self.source.validate_key(key)
    }
}
