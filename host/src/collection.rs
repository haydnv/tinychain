//! A [`Collection`] such as a [`BTree`] or [`Table`].

/// The `Collection` enum used in `State::Collection`.
use std::fmt;

use async_trait::async_trait;
use bytes::Bytes;
use destream::{de, en};
use futures::{TryFutureExt, TryStreamExt};
use log::debug;
use sha2::{Digest, Sha256};

use tc_btree::{BTreeInstance, BTreeView};
use tc_error::*;
use tc_table::{TableInstance, TableView};
use tc_transact::fs::Dir;
use tc_transact::{IntoView, Transaction, TxnId};
use tcgeneric::{
    path_label, Class, Instance, NativeClass, PathLabel, PathSegment, TCPath, TCPathBuf,
};

use crate::fs;
use crate::txn::Txn;

pub use tc_btree::BTreeType;
pub use tc_table::TableType;

pub type BTree = tc_btree::BTree<fs::File<tc_btree::Node>, fs::Dir, Txn>;
pub type BTreeFile = tc_btree::BTreeFile<fs::File<tc_btree::Node>, fs::Dir, Txn>;
pub type Table = tc_table::Table<fs::File<tc_btree::Node>, fs::Dir, Txn>;
pub type TableIndex = tc_table::TableIndex<fs::File<tc_btree::Node>, fs::Dir, Txn>;

const PREFIX: PathLabel = path_label(&["state", "collection"]);

/// The [`Class`] of a [`Collection`].
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum CollectionType {
    BTree(BTreeType),
    Table(TableType),
}

impl Class for CollectionType {}

impl NativeClass for CollectionType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        debug!("CollectionType::from_path {}", TCPath::from(path));

        if path.len() > 2 && &path[0..2] == &PREFIX[..] {
            match path[2].as_str() {
                "btree" => BTreeType::from_path(path).map(Self::BTree),
                "table" => TableType::from_path(path).map(Self::Table),
                _ => None,
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        match self {
            Self::BTree(btree) => btree.path(),
            Self::Table(table) => table.path(),
        }
    }
}

impl From<BTreeType> for CollectionType {
    fn from(btt: BTreeType) -> Self {
        Self::BTree(btt)
    }
}

impl From<TableType> for CollectionType {
    fn from(tt: TableType) -> Self {
        Self::Table(tt)
    }
}

impl fmt::Display for CollectionType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(btt) => fmt::Display::fmt(btt, f),
            Self::Table(tt) => fmt::Display::fmt(tt, f),
        }
    }
}

/// A stateful, transaction-aware [`Collection`], such as a [`BTree`] or [`Table`].
#[derive(Clone)]
pub enum Collection {
    BTree(BTree),
    Table(Table),
}

impl Collection {
    #[allow(dead_code)]
    async fn hash(self, txn_id: TxnId) -> TCResult<Bytes> {
        // TODO: can this be consolidated with transact::fs::Block::hash?

        let mut hasher = Sha256::default();

        async fn hash_chunks<'en, T: en::IntoStream<'en> + 'en>(
            hasher: &mut Sha256,
            data: T,
        ) -> TCResult<()> {
            let mut data = destream_json::en::encode(data).map_err(TCError::internal)?;
            while let Some(chunk) = data.try_next().map_err(TCError::internal).await? {
                hasher.update(&chunk);
            }

            Ok(())
        }

        match self {
            Self::BTree(btree) => {
                let mut keys = btree.keys(txn_id).await?;
                while let Some(key) = keys.try_next().await? {
                    hash_chunks(&mut hasher, key).await?;
                }
            }
            Self::Table(table) => {
                let mut rows = table.rows(txn_id).await?;
                while let Some(row) = rows.try_next().await? {
                    hash_chunks(&mut hasher, row).await?;
                }
            }
        }

        let digest = hasher.finalize();
        Ok(Bytes::from(digest.to_vec()))
    }
}

impl Instance for Collection {
    type Class = CollectionType;

    fn class(&self) -> Self::Class {
        match self {
            Self::BTree(btree) => CollectionType::BTree(btree.class()),
            Self::Table(table) => CollectionType::Table(table.class()),
        }
    }
}

impl From<BTree> for Collection {
    fn from(btree: BTree) -> Self {
        Self::BTree(btree)
    }
}

impl From<Table> for Collection {
    fn from(table: Table) -> Self {
        Self::Table(table)
    }
}

/// A [`de::Visitor`] used to deserialize a [`Collection`].
pub struct CollectionVisitor {
    txn: Txn,
}

impl CollectionVisitor {
    pub fn new(txn: Txn) -> Self {
        Self { txn }
    }

    pub async fn visit_map_value<A: de::MapAccess>(
        self,
        class: CollectionType,
        access: &mut A,
    ) -> Result<Collection, A::Error> {
        match class {
            CollectionType::BTree(_) => {
                let file = self
                    .txn
                    .context()
                    .create_file_tmp(*self.txn.id(), BTreeType::default())
                    .map_err(de::Error::custom)
                    .await?;

                access
                    .next_value((self.txn.clone(), file))
                    .map_ok(Collection::BTree)
                    .await
            }
            CollectionType::Table(_) => access.next_value(self.txn).map_ok(Collection::Table).await,
        }
    }
}

#[async_trait]
impl de::Visitor for CollectionVisitor {
    type Value = Collection;

    fn expecting() -> &'static str {
        "a Collection"
    }

    async fn visit_map<A: de::MapAccess>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let classpath = map
            .next_key::<TCPathBuf>(())
            .await?
            .ok_or_else(|| de::Error::custom("expected a Collection type"))?;

        let class = CollectionType::from_path(&classpath)
            .ok_or_else(|| de::Error::invalid_value(classpath, "a Collection type"))?;

        self.visit_map_value(class, &mut map).await
    }
}

#[async_trait]
impl de::FromStream for Collection {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_map(CollectionVisitor { txn }).await
    }
}

#[async_trait]
impl<'en> IntoView<'en, fs::Dir> for Collection {
    type Txn = Txn;
    type View = CollectionView<'en>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        match self {
            Self::BTree(btree) => btree.into_view(txn).map_ok(CollectionView::BTree).await,
            Self::Table(table) => table.into_view(txn).map_ok(CollectionView::Table).await,
        }
    }
}

impl fmt::Display for Collection {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(btree) => fmt::Display::fmt(btree, f),
            Self::Table(table) => fmt::Display::fmt(table, f),
        }
    }
}

/// A view of a [`Collection`] within a single `Transaction`, used for serialization.
pub enum CollectionView<'en> {
    BTree(BTreeView<'en>),
    Table(TableView<'en>),
}

impl<'en> en::IntoStream<'en> for CollectionView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        use destream::en::EncodeMap;

        let mut map = encoder.encode_map(Some(1))?;
        match self {
            Self::BTree(btree) => map.encode_entry(BTreeType::default().path(), btree),
            Self::Table(table) => map.encode_entry(TableType::default().path(), table),
        }?;
        map.end()
    }
}
