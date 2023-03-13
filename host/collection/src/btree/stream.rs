use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use async_trait::async_trait;
use destream::en;
use futures::{Stream, TryFutureExt};
use safecast::AsType;

use tc_error::*;
use tc_transact::{IntoView, Transaction};
use tcgeneric::{TCBoxTryStream, ThreadSafe};

use super::{BTree, BTreeInstance, Key, Node, Range};

type PermitRead = tc_transact::lock::PermitRead<Arc<Range>>;

/// A stream over a range of keys in a `BTree`
pub struct Keys<'a> {
    #[allow(unused)]
    permit: PermitRead,
    keys: TCBoxTryStream<'a, Key>,
}

impl<'a> Keys<'a> {
    pub(super) fn new(permit: PermitRead, keys: TCBoxTryStream<'a, Key>) -> Self {
        Self { permit, keys }
    }
}

impl<'a> Stream for Keys<'a> {
    type Item = TCResult<Key>;

    fn poll_next(mut self: Pin<&mut Self>, cxt: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.keys).poll_next(cxt)
    }
}

pub struct BTreeView<'en> {
    keys: Keys<'en>,
}

impl<'en> en::IntoStream<'en> for BTreeView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_seq_stream(self.keys)
    }
}

impl<'en> From<Keys<'en>> for BTreeView<'en> {
    fn from(keys: Keys<'en>) -> Self {
        Self { keys }
    }
}

#[async_trait]
impl<'en, Txn, FE> IntoView<'en, FE> for BTree<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type Txn = Txn;
    type View = BTreeView<'en>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        match self {
            Self::File(file) => file.keys(*txn.id()).map_ok(BTreeView::from).await,
            Self::Slice(slice) => slice.keys(*txn.id()).map_ok(BTreeView::from).await,
        }
    }
}
