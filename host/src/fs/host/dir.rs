use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::path::PathBuf;

use uplock::RwLock;

use tc_error::*;
use tcgeneric::PathSegment;

use crate::chain::ChainBlock;
use crate::scalar::Value;

use super::file::File;
use super::{fs_path, BlockData, Cache};

const ERR_ALREADY_EXISTS: &str = "the filesystem already has an entry called";

#[derive(Clone)]
pub enum DirEntry {
    Dir(Dir),
    File(FileEntry),
}

impl fmt::Display for DirEntry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Dir(_) => f.write_str("(a directory)"),
            Self::File(file) => fmt::Display::fmt(file, f),
        }
    }
}

#[derive(Clone)]
pub enum FileEntry {
    Chain(File<ChainBlock>),
    Value(File<Value>),
}

impl From<File<ChainBlock>> for FileEntry {
    fn from(file: File<ChainBlock>) -> Self {
        Self::Chain(file)
    }
}

impl TryFrom<FileEntry> for File<ChainBlock> {
    type Error = TCError;

    fn try_from(file: FileEntry) -> TCResult<Self> {
        match file {
            FileEntry::Chain(file) => Ok(file),
            other => Err(TCError::bad_request(
                "expected File<Chain> but found",
                other,
            )),
        }
    }
}

impl From<File<Value>> for FileEntry {
    fn from(file: File<Value>) -> Self {
        Self::Value(file)
    }
}

impl TryFrom<FileEntry> for File<Value> {
    type Error = TCError;

    fn try_from(file: FileEntry) -> TCResult<Self> {
        match file {
            FileEntry::Value(file) => Ok(file),
            other => Err(TCError::bad_request(
                "expected File<Value> but found",
                other,
            )),
        }
    }
}

impl fmt::Display for FileEntry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Self::Chain(_) => "File<Chain>",
            Self::Value(_) => "File<Value>",
        })
    }
}

#[derive(Clone)]
pub struct Dir {
    path: PathBuf,
    cache: Cache,
    contents: RwLock<HashMap<PathSegment, DirEntry>>,
}

impl Dir {
    pub async fn create_dir(&self, name: PathSegment) -> TCResult<Dir> {
        let mut contents = self.contents.write().await;
        if contents.contains_key(&name) {
            return Err(TCError::bad_request(ERR_ALREADY_EXISTS, name));
        }

        let subdir = Dir {
            path: fs_path(&self.path, &name),
            cache: self.cache.clone(),
            contents: RwLock::new(HashMap::new()),
        };

        contents.insert(name, DirEntry::Dir(subdir.clone()));
        Ok(subdir)
    }

    pub async fn create_file<B: BlockData>(&self, name: PathSegment) -> TCResult<File<B>>
    where
        FileEntry: From<File<B>>,
    {
        let mut contents = self.contents.write().await;
        if contents.contains_key(&name) {
            return Err(TCError::bad_request(ERR_ALREADY_EXISTS, name));
        }

        let file = File::create(fs_path(&self.path, &name), self.cache.clone());
        contents.insert(name, DirEntry::File(file.clone().into()));
        Ok(file)
    }

    pub async fn contains(&self, name: &PathSegment) -> bool {
        let contents = self.contents.read().await;
        contents.contains_key(name)
    }

    pub async fn get_dir(&self, name: &PathSegment) -> TCResult<Option<Dir>> {
        let contents = self.contents.read().await;
        match contents.get(name) {
            Some(DirEntry::Dir(dir)) => Ok(Some(dir.clone())),
            Some(other) => Err(TCError::bad_request("not a Dir", other)),
            None => Ok(None),
        }
    }

    pub async fn get_file<B: BlockData>(&self, name: &PathSegment) -> TCResult<Option<File<B>>>
    where
        File<B>: TryFrom<FileEntry, Error = TCError>,
    {
        let contents = self.contents.read().await;
        match contents.get(name) {
            Some(DirEntry::File(file)) => file.clone().try_into().map(Some),
            Some(other) => Err(TCError::bad_request("not a Dir", other)),
            None => Ok(None),
        }
    }
}
