use std::fmt;

use tcgeneric::{path_label, Class, Instance, NativeClass, PathLabel, PathSegment, TCPathBuf};

mod bounds;
mod index;
mod schema;
mod view;

use index::*;
use view::*;

pub use bounds::*;
pub use index::TableIndex;
pub use schema::*;

const PATH: PathLabel = path_label(&["state", "collection", "table"]);

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum TableType {
    Index,
    ReadOnly,
    Table,
    Aggregate,
    IndexSlice,
    Limit,
    Merge,
    Selection,
    TableSlice,
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
            Self::Index => write!(f, "type Index"),
            Self::ReadOnly => write!(f, "type Index (read-only)"),
            Self::Table => write!(f, "type Table"),
            Self::Aggregate => write!(f, "type Aggregate"),
            Self::IndexSlice => write!(f, "type Index slice"),
            Self::Limit => write!(f, "type Limit selection"),
            Self::Merge => write!(f, "type Merge selection"),
            Self::Selection => write!(f, "type Column selection"),
            Self::TableSlice => write!(f, "type Table slice"),
        }
    }
}

#[derive(Clone)]
pub enum Table {
    Index(Index),
    ROIndex(ReadOnly),
    Table(TableIndex),
    Aggregate(Aggregate),
    IndexSlice(IndexSlice),
    Limit(Limited),
    Merge(Merged),
    Selection(Selection),
    TableSlice(TableSlice),
}

impl Instance for Table {
    type Class = TableType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Index(_) => TableType::Index,
            Self::ROIndex(_) => TableType::ReadOnly,
            Self::Table(_) => TableType::Table,
            Self::Aggregate(_) => TableType::Aggregate,
            Self::IndexSlice(_) => TableType::IndexSlice,
            Self::Limit(_) => TableType::Limit,
            Self::Merge(_) => TableType::Merge,
            Self::Selection(_) => TableType::Selection,
            Self::TableSlice(_) => TableType::TableSlice,
        }
    }
}

impl fmt::Display for Table {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "an instance of {}", self.class())
    }
}
