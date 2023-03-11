use std::marker::PhantomData;

use async_trait::async_trait;
use destream::{de, en};
use safecast::AsType;

use tc_error::*;
use tc_transact::{IntoView, Transaction};

use super::{BTree, Node};

pub struct BTreeView<'en> {
    phantom: PhantomData<&'en ()>,
}

impl<'en> Clone for BTreeView<'en> {
    fn clone(&self) -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<'en> en::IntoStream<'en> for BTreeView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        todo!()
    }
}

#[async_trait]
impl<'en, Txn, FE> IntoView<'en, FE> for BTree<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + Send + Sync + 'en,
{
    type Txn = Txn;
    type View = BTreeView<'en>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        Err(not_implemented!("BTreeFile::into_view"))
    }
}
