pub mod block;
pub mod chain;
pub mod file;
mod history;

pub const RECORD_DELIMITER: char = 30 as char;
pub const GROUP_DELIMITER: char = 29 as char;

pub type History<O> = history::History<O>;
