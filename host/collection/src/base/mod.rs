#[cfg(all(feature = "btree", not(feature = "table")))]
mod btree;
#[cfg(not(feature = "btree"))]
mod mock;
#[cfg(all(feature = "table", not(feature = "tensor")))]
mod table;
#[cfg(feature = "tensor")]
mod tensor;

#[cfg(all(feature = "btree", not(feature = "table")))]
pub use btree::{CollectionBase, CollectionVisitor};
#[cfg(not(feature = "btree"))]
pub use mock::{CollectionBase, CollectionVisitor};
#[cfg(all(feature = "table", not(feature = "tensor")))]
pub use table::{CollectionBase, CollectionVisitor};
#[cfg(feature = "tensor")]
pub use tensor::{CollectionBase, CollectionVisitor};
