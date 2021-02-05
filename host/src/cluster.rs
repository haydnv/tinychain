use std::fmt;

use generic::{TCPathBuf, PathSegment};

#[derive(Clone)]
pub struct Cluster {
    path: TCPathBuf,
}

impl Cluster {
    pub fn path(&'_ self) -> &'_ [PathSegment] {
        &self.path
    }
}

impl fmt::Display for Cluster {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cluster at {}", self.path)
    }
}
