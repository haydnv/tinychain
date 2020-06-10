pub mod archive;
pub mod chain;
mod dir;
mod file;

pub const RECORD_DELIMITER: char = 30 as char;
pub const GROUP_DELIMITER: char = 29 as char;

pub type Dir = dir::Dir;
pub type File = file::File;
