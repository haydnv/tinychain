use async_trait::async_trait;

use tc_error::{TCError, TCResult};
use tc_transact::fs::{Dir, File};
use tc_transact::{Transaction, TxnId};
use tc_value::ValueCollator;
use tcgeneric::{Instance, TCTryStream};

use super::{
    validate_range, BTree, BTreeFile, BTreeInstance, BTreeType, Key, Node, Range, RowSchema,
};
use futures::{TryFutureExt, TryStreamExt};

#[derive(Clone)]
pub struct BTreeSlice<F, D, T> {
    source: BTreeFile<F, D, T>,
    range: Range,
    reverse: bool,
}

impl<F, D, T> BTreeSlice<F, D, T>
where
    BTreeFile<F, D, T>: Clone + Send + Sync,
{
    pub fn new(source: BTree<F, D, T>, range: Range, reverse: bool) -> BTreeSlice<F, D, T> {
        match source {
            BTree::File(tree) => Self {
                source: tree,
                range,
                reverse,
            },

            BTree::Slice(view) => {
                let source = view.source.clone();
                let reverse = view.reverse ^ reverse;

                if range == Range::default() {
                    Self {
                        source,
                        range: view.range,
                        reverse,
                    }
                } else {
                    // TODO: validate that the current range contains the requested range

                    Self {
                        source,
                        range,
                        reverse,
                    }
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
impl<F: File<Node>, D: Dir, T: Transaction<D>> BTreeInstance for BTreeSlice<F, D, T>
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
            Ok(Self::new(BTree::Slice(self), range, reverse))
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

    async fn delete(&self, txn_id: TxnId) -> TCResult<()> {
        self.source.delete_range(txn_id, &self.range).await
    }

    async fn insert(&self, txn_id: TxnId, key: Key) -> TCResult<()> {
        self.source.insert(txn_id, key).await
    }

    async fn keys<'a>(self, txn_id: TxnId) -> TCResult<TCTryStream<'a, Key>> {
        self.source
            .rows_in_range(txn_id, self.range, self.reverse)
            .await
    }
}
