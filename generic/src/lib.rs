pub mod id;
pub mod map;
pub mod tuple;

pub use id::*;
pub use map::*;
pub use tuple::*;

pub trait Class: Sized {
    type Instance;
}

pub trait NativeClass: Class {
    fn from_path(path: &[PathSegment]) -> Option<Self>;

    fn path(&self) -> TCPathBuf;
}

pub trait Instance {
    type Class: Class;

    fn class(&self) -> Self::Class;
}
