use std::fmt;

use tcgeneric::{path_label, Class, Instance, NativeClass, PathLabel, PathSegment, TCPathBuf};

const PREFIX: PathLabel = path_label(&["state", "collection"]);

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum CollectionType {}

impl Class for CollectionType {
    type Instance = Collection;
}

impl NativeClass for CollectionType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if &path[0..2] == &PREFIX[..] {
            todo!()
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        todo!()
    }
}

impl fmt::Display for CollectionType {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Clone)]
pub enum Collection {}

impl Instance for Collection {
    type Class = CollectionType;

    fn class(&self) -> Self::Class {
        todo!()
    }
}

impl fmt::Display for Collection {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}
