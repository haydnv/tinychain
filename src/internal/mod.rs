pub mod archive;
mod cache;
pub mod chain;
mod lock;

// TODO: DELETE
mod dir;
// TODO: DELETE
mod file;

// TODO: DELETE
pub const RECORD_DELIMITER: char = 30 as char;

// TODO: DELETE
pub const GROUP_DELIMITER: char = 29 as char;

pub type BlockId = file::BlockId;
pub type Dir = dir::Dir;
pub type File = file::File;
