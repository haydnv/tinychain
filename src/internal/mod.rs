pub mod block;
pub mod cache;
mod chain;
mod directory;
pub mod file;

pub const RECORD_DELIMITER: char = 30 as char;
pub const GROUP_DELIMITER: char = 29 as char;

pub type Chain = chain::Chain;
