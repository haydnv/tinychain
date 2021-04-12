use tc_value::ValueCollator;
use tcgeneric::Instance;

use super::{BTree, BTreeFile, BTreeInstance, BTreeType, Range, RowSchema};

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

impl<F, D, T> BTreeInstance for BTreeSlice<F, D, T>
where
    BTreeFile<F, D, T>: Clone + Send + Sync,
{
    type Slice = Self;

    fn collator(&'_ self) -> &'_ ValueCollator {
        self.source.collator()
    }

    fn schema(&'_ self) -> &'_ RowSchema {
        self.source.schema()
    }

    fn slice(self, range: Range, reverse: bool) -> Self::Slice {
        Self::new(BTree::Slice(self), range, reverse)
    }
}
