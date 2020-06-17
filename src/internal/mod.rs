pub mod archive;
pub mod cache;
pub mod chain;
pub mod lock;

// TODO: DELETE
mod file;

// TODO: DELETE
pub const RECORD_DELIMITER: char = 30 as char;

// TODO: DELETE
pub const GROUP_DELIMITER: char = 29 as char;

pub type BlockId = file::BlockId;
pub type File = file::File;
