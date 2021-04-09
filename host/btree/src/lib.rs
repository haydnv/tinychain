use std::cmp::Ordering;
use std::fmt;

use collate::Collate;
use log::debug;
use safecast::{Match, TryCastFrom, TryCastInto};

use tc_error::*;
use tc_value::{Bound, NumberType, Range, Value, ValueCollator, ValueType};
use tcgeneric::*;

#[allow(dead_code)]
mod file;
mod slice;

const PREFIX: PathLabel = path_label(&["state", "collection", "btree"]);

pub use file::{BTreeFile, Node};
pub use slice::BTreeSlice;

pub type Key = Vec<Value>;

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

#[derive(Clone, Eq, PartialEq)]
pub struct BTreeRange(Vec<Bound>, Vec<Bound>);

impl BTreeRange {
    pub fn contains(&self, other: &BTreeRange, collator: &ValueCollator) -> bool {
        use Bound::*;
        use Ordering::*;

        for (outer, inner) in self.0.iter().zip(&other.0) {
            match (outer, inner) {
                (Unbounded, _) => {}
                (_, Unbounded) => return false,
                (Ex(o), Ex(i)) if collator.compare(&o, &i) == Greater => return false,
                (In(o), In(i)) if collator.compare(&o, &i) == Greater => return false,
                (In(o), Ex(i)) if collator.compare(&o, &i) == Greater => return false,
                (Ex(o), In(i)) if collator.compare(&o, &i) != Less => return false,
                _ => {}
            }
        }

        for (outer, inner) in self.1.iter().zip(&other.1) {
            match (outer, inner) {
                (Unbounded, _) => {}
                (_, Unbounded) => return false,
                (Ex(o), Ex(i)) if collator.compare(&o, &i) == Less => return false,
                (In(o), In(i)) if collator.compare(&o, &i) == Less => return false,
                (In(o), Ex(i)) if collator.compare(&o, &i) == Less => return false,
                (Ex(o), In(i)) if collator.compare(&o, &i) != Greater => return false,
                _ => {}
            }
        }

        true
    }

    pub fn is_key(&self, schema: &[Column]) -> bool {
        self.0.len() == self.1.len()
            && self.0.len() == schema.len()
            && self.0.iter().zip(self.1.iter()).all(|(l, r)| l == r)
    }

    pub fn start(&'_ self) -> &'_ [Bound] {
        &self.0
    }

    pub fn end(&'_ self) -> &'_ [Bound] {
        &self.1
    }
}

pub fn validate_range<T: fmt::Display>(range: T, schema: &[Column]) -> TCResult<BTreeRange>
where
    BTreeRange: TryCastFrom<T>,
{
    use Bound::*;

    let range =
        BTreeRange::try_cast_from(range, |v| TCError::bad_request("Invalid BTreeRange", v))?;

    let cast = |(bound, column): (Bound, &Column)| {
        let value = match bound {
            Unbounded => Unbounded,
            In(value) => In(column.dtype().try_cast(value)?),
            Ex(value) => Ex(column.dtype().try_cast(value)?),
        };
        Ok(value)
    };

    let cast_range = |range: Vec<Bound>| {
        range
            .into_iter()
            .zip(schema)
            .map(cast)
            .collect::<TCResult<Vec<Bound>>>()
    };

    let start = cast_range(range.0)?;
    let end = cast_range(range.1)?;
    Ok(BTreeRange(start, end))
}

impl Default for BTreeRange {
    fn default() -> Self {
        Self(vec![], vec![])
    }
}

impl From<Key> for BTreeRange {
    fn from(key: Key) -> Self {
        let start = key.iter().cloned().map(Bound::In).collect();
        let end = key.into_iter().map(Bound::In).collect();
        Self(start, end)
    }
}

impl From<(Vec<Bound>, Vec<Bound>)> for BTreeRange {
    fn from(params: (Vec<Bound>, Vec<Bound>)) -> Self {
        Self(params.0, params.1)
    }
}

impl From<Vec<Range>> for BTreeRange {
    fn from(range: Vec<Range>) -> Self {
        Self::from(range.into_iter().map(Range::into_inner).unzip())
    }
}

impl TryCastFrom<Value> for BTreeRange {
    fn can_cast_from(value: &Value) -> bool {
        if value == &Value::None || Key::can_cast_from(value) {
            true
        } else if let Value::Tuple(tuple) = value {
            tuple.iter().all(|v| v.is_none())
        } else {
            false
        }
    }

    fn opt_cast_from(value: Value) -> Option<BTreeRange> {
        if value == Value::None {
            Some(BTreeRange::default())
        } else if let Value::Tuple(tuple) = value {
            if tuple.iter().all(|v| v.is_none()) {
                Some(BTreeRange::default())
            } else {
                None
            }
        } else {
            Key::opt_cast_from(value).map(BTreeRange::from)
        }
    }
}

impl fmt::Display for BTreeRange {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0.is_empty() && self.1.is_empty() {
            return write!(f, "BTreeRange::default");
        }

        let to_str = |bounds: &[Bound]| {
            bounds
                .iter()
                .map(|bound| bound.to_string())
                .collect::<Vec<String>>()
                .join(", ")
        };

        write!(
            f,
            "BTreeRange: (from: {}, to: {})",
            to_str(&self.0),
            to_str(&self.1)
        )
    }
}

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
    Slice(BTreeSlice),
}

impl<F: Send + Sync, D: Send + Sync, T: Send + Sync> Instance for BTree<F, D, T> {
    type Class = BTreeType;

    fn class(&self) -> BTreeType {
        match self {
            Self::File(_) => BTreeType::File,
            Self::Slice(_) => BTreeType::Slice,
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
