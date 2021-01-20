pub mod id;
pub mod map;
pub mod tuple;

pub use id::*;
pub use map::*;
pub use tuple::*;

pub trait Class {
    type Instance;

    fn path(&self) -> TCPathBuf;
}

pub trait Instance {
    type Class: Class;

    fn class(&self) -> Self::Class;
}
