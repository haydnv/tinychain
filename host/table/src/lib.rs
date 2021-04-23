use std::fmt;

use tcgeneric::{path_label, Class, Instance, NativeClass, PathLabel, PathSegment, TCPathBuf};

mod schema;

pub use schema::*;

const PATH: PathLabel = path_label(&["state", "collection", "table"]);

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum TableType {
    Table,
}

impl Class for TableType {}

impl NativeClass for TableType {
    // these functions are only used for serialization, and only a base table can be deserialized

    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() == 3 && &path[..] == &PATH[..] {
            Some(Self::Table)
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        PATH.into()
    }
}

impl fmt::Display for TableType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Table => f.write_str("type Table"),
        }
    }
}

#[derive(Clone)]
pub enum Table {}

impl Instance for Table {
    type Class = TableType;

    fn class(&self) -> Self::Class {
        // TODO: match self
        TableType::Table
    }
}

impl fmt::Display for Table {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // TODO: match self
        f.write_str("a Table")
    }
}
