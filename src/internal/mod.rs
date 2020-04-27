mod chain;
mod fs;

pub const DELIMITER: char = 30 as char;
pub const GROUP_DELIMITER: char = 29 as char;

pub type Chain = chain::Chain;
pub type FsDir = fs::Dir;

pub mod cache;
mod directory;
pub mod file;
