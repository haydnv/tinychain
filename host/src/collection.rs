use std::convert::TryInto;
use std::fmt;

use async_trait::async_trait;
use destream::{de, en};
use futures::TryFutureExt;
use log::debug;

use tc_btree::BTreeView;
use tc_error::*;
use tc_table::{Table, TableType};
use tc_transact::fs::Dir;
use tc_transact::{IntoView, Transaction};
use tcgeneric::{
    path_label, Class, Instance, NativeClass, PathLabel, PathSegment, TCPath, TCPathBuf,
};

use crate::fs;
use crate::txn::Txn;

const PREFIX: PathLabel = path_label(&["state", "collection"]);

pub use tc_btree::BTreeType;

pub type BTree = tc_btree::BTree<fs::File<tc_btree::Node>, fs::Dir, Txn>;
pub type BTreeFile = tc_btree::BTreeFile<fs::File<tc_btree::Node>, fs::Dir, Txn>;

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

impl fmt::Display for CollectionType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(btt) => fmt::Display::fmt(btt, f),
            Self::Table(tt) => fmt::Display::fmt(tt, f),
        }
    }
}

#[derive(Clone)]
pub enum Collection {
    BTree(BTree),
    Table(Table),
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
                    .create_file_tmp(*self.txn.id(), BTreeType::default().into())
                    .map_err(de::Error::custom)
                    .await?;

                let file = file.try_into().map_err(de::Error::custom)?;

                access
                    .next_value((self.txn.clone(), file))
                    .map_ok(Collection::BTree)
                    .await
            }
            CollectionType::Table(_) => {
                unimplemented!()
            }
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
            Self::Table(_table) => unimplemented!(),
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

pub enum CollectionView<'en> {
    BTree(BTreeView<'en>),
}

impl<'en> en::IntoStream<'en> for CollectionView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::BTree(btree) => btree.into_stream(encoder),
        }
    }
}
