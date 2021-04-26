use tc_btree::Node;
use tc_error::*;
use tc_transact::fs::{Dir, File};
use tc_transact::Transaction;
use tcgeneric::{Id, Instance};

use super::{IndexSlice, Table, TableInstance, TableType};

#[derive(Clone)]
pub struct Index;

#[derive(Clone)]
pub struct ReadOnly {
    index: IndexSlice,
}

impl ReadOnly {
    pub async fn copy_from<
        F: File<Node>,
        D: Dir,
        Txn: Transaction<D>,
        T: TableInstance<F, D, Txn>,
    >(
        _source: T,
        _txn: Txn,
        _key_columns: Option<Vec<Id>>,
    ) -> TCResult<ReadOnly> {
        todo!()
    }
}

impl Instance for ReadOnly {
    type Class = TableType;

    fn class(&self) -> Self::Class {
        Self::Class::ReadOnly
    }
}

#[derive(Clone)]
pub struct TableIndex<F, D, Txn> {
    file: F,
    dir: D,
    txn: Txn,
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>> From<TableIndex<F, D, Txn>> for Table<F, D, Txn> {
    fn from(table: TableIndex<F, D, Txn>) -> Self {
        Self::Table(table)
    }
}
