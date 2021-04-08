use std::fmt;

use tcgeneric::{
    label, path_label, Class, Instance, NativeClass, PathLabel, PathSegment, TCPathBuf,
};

mod file;
mod slice;

pub use file::BTreeFile;
pub use slice::BTreeSlice;

const PREFIX: PathLabel = path_label(&["state", "collection", "btree"]);

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum BTreeType {
    File,
    Slice,
}

impl Class for BTreeType {
    type Instance = BTree;
}

impl NativeClass for BTreeType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() == 4 && &path[0..3] == &PREFIX[..] {
            match path[3].as_str() {
                "file" => Some(Self::File),
                "slice" => Some(Self::Slice),
                _ => None,
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        let path = TCPathBuf::from(PREFIX);
        path.append(label(match self {
            Self::File => "file",
            Self::Slice => "slice",
        }))
    }
}

impl fmt::Display for BTreeType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::File => f.write_str("type BTree"),
            Self::Slice => f.write_str("type BTreeSlice"),
        }
    }
}

#[derive(Clone)]
pub enum BTree {
    File(BTreeFile),
    Slice(BTreeSlice),
}

impl Instance for BTree {
    type Class = BTreeType;

    fn class(&self) -> BTreeType {
        match self {
            Self::File(_) => BTreeType::File,
            Self::Slice(_) => BTreeType::Slice,
        }
    }
}

impl fmt::Display for BTree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Self::File(_) => "a BTree",
            Self::Slice(_) => "a BTree slice",
        })
    }
}
