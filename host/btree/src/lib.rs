use std::fmt;
use std::ops::Bound;

use async_trait::async_trait;
use log::debug;
use safecast::{Match, TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::TxnId;
use tc_value::{NumberType, Value, ValueCollator, ValueType};
use tcgeneric::*;

#[allow(dead_code)]
mod file;
mod slice;

const PREFIX: PathLabel = path_label(&["state", "collection", "btree"]);

pub use file::{BTreeFile, Node};
pub use slice::BTreeSlice;

pub type Key = Vec<Value>;
pub type Range = collate::Range<Value, Key>;

#[async_trait]
pub trait BTreeInstance: Instance {
    type Slice: BTreeInstance;

    fn collator(&self) -> &ValueCollator;

    fn schema(&self) -> &RowSchema;

    fn slice(self, range: Range, reverse: bool) -> Self::Slice;

    async fn delete(&self, txn_id: TxnId) -> TCResult<()>;

    async fn insert(&self, txn_id: TxnId, key: Key) -> TCResult<()>;

    async fn rows(self, txn_id: TxnId) -> TCResult<TCTryStream<'static, Key>>;
}

#[derive(Clone, PartialEq)]
pub struct Column {
    name: Id,
    dtype: ValueType,
    max_len: Option<usize>,
}

impl Column {
    pub fn name(&'_ self) -> &'_ Id {
        &self.name
    }

    pub fn dtype(&self) -> ValueType {
        self.dtype
    }

    pub fn max_len(&'_ self) -> &'_ Option<usize> {
        &self.max_len
    }
}

impl<I: Into<Id>> From<(I, NumberType)> for Column {
    fn from(column: (I, NumberType)) -> Column {
        let (name, dtype) = column;
        let name: Id = name.into();
        let dtype: ValueType = dtype.into();
        let max_len = None;

        Column {
            name,
            dtype,
            max_len,
        }
    }
}

impl From<(Id, ValueType)> for Column {
    fn from(column: (Id, ValueType)) -> Column {
        let (name, dtype) = column;
        let max_len = None;

        Column {
            name,
            dtype,
            max_len,
        }
    }
}

impl From<(Id, ValueType, usize)> for Column {
    fn from(column: (Id, ValueType, usize)) -> Column {
        let (name, dtype, size) = column;
        let max_len = Some(size);

        Column {
            name,
            dtype,
            max_len,
        }
    }
}

impl TryCastFrom<Value> for Column {
    fn can_cast_from(value: &Value) -> bool {
        debug!("Column::can_cast_from {}?", value);

        value.matches::<(Id, ValueType)>() || value.matches::<(Id, ValueType, u64)>()
    }

    fn opt_cast_from(value: Value) -> Option<Column> {
        if value.matches::<(Id, ValueType)>() {
            let (name, dtype) = value.opt_cast_into().unwrap();

            Some(Column {
                name,
                dtype,
                max_len: None,
            })
        } else if value.matches::<(Id, ValueType, u64)>() {
            let (name, dtype, max_len) = value.opt_cast_into().unwrap();

            Some(Column {
                name,
                dtype,
                max_len: Some(max_len),
            })
        } else {
            None
        }
    }
}

impl<'a> From<&'a Column> for (&'a Id, ValueType) {
    fn from(col: &'a Column) -> (&'a Id, ValueType) {
        (&col.name, col.dtype)
    }
}

impl fmt::Display for Column {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.max_len {
            Some(max_len) => write!(f, "{}: {}({})", self.name, self.dtype, max_len),
            None => write!(f, "{}: {}", self.name, self.dtype),
        }
    }
}

pub type RowSchema = Vec<Column>;

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum BTreeType {
    File,
    Slice,
}

impl Class for BTreeType {}

impl NativeClass for BTreeType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() == 4 && &path[0..3] == &PREFIX[..] {
            match path[3].as_str() {
                "file" => Some(Self::File),
                "slice" => Some(Self::Slice),
                _ => None,
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        let path = TCPathBuf::from(PREFIX);
        path.append(label(match self {
            Self::File => "file",
            Self::Slice => "slice",
        }))
    }
}

impl fmt::Display for BTreeType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::File => f.write_str("type BTree"),
            Self::Slice => f.write_str("type BTreeSlice"),
        }
    }
}

#[derive(Clone)]
pub enum BTree<F, D, T> {
    File(BTreeFile<F, D, T>),
    Slice(BTreeSlice<F, D, T>),
}

impl<F: Send + Sync, D: Send + Sync, T: Send + Sync> Instance for BTree<F, D, T> {
    type Class = BTreeType;

    fn class(&self) -> BTreeType {
        match self {
            Self::File(file) => file.class(),
            Self::Slice(slice) => slice.class(),
        }
    }
}

impl<F, D, T> fmt::Display for BTree<F, D, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Self::File(_) => "a BTree",
            Self::Slice(_) => "a BTree slice",
        })
    }
}

#[inline]
pub fn validate_range(range: Range, schema: &[Column]) -> TCResult<Range> {
    if range.len() > schema.len() {
        return Err(TCError::bad_request(
            "too many columns in range",
            range.len(),
        ));
    }

    let (input_prefix, start, end) = range.into_inner();

    let mut prefix = Vec::with_capacity(input_prefix.len());
    for (value, column) in input_prefix.into_iter().zip(schema) {
        let value = column.dtype.try_cast(value)?;
        prefix.push(value);
    }

    let dtype = schema.get(prefix.len()).unwrap().dtype;
    let validate_bound = |bound| match bound {
        Bound::Unbounded => Ok(Bound::Unbounded),
        Bound::Included(value) => dtype.try_cast(value).map(Bound::Included),
        Bound::Excluded(value) => dtype.try_cast(value).map(Bound::Excluded),
    };

    let start = validate_bound(start)?;
    let end = validate_bound(end)?;

    Ok((prefix, start, end).into())
}
