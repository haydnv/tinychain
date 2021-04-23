use std::fmt;

use tcgeneric::{path_label, Class, Instance, NativeClass, PathLabel, PathSegment, TCPathBuf};

const PATH: PathLabel = path_label(&["state", "collection", "table"]);

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum TableType {
    Table,
}

impl Class for TableType {}

impl NativeClass for TableType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        // This path is only used for serialization, and only a base table can be deserialized
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
pub struct Table;

impl Instance for Table {
    type Class = TableType;

    fn class(&self) -> Self::Class {
        TableType::Table
    }
}
