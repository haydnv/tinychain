use std::fmt;
use std::ops::Bound;

use async_hash::{Digest, Hash, Output};
use async_trait::async_trait;
use destream::{de, en};
use safecast::{CastFrom, Match, TryCastFrom, TryCastInto};

use tc_error::{bad_request, TCError, TCResult};
use tc_value::{NumberType, Value, ValueType};
use tcgeneric::{Id, NativeClass};

use super::{Key, Range};

const MEGA: usize = 1_000_000;
const MIN_ORDER: usize = 4;
const TERA: usize = 1_000_000_000_000;
const UUID_SIZE: usize = 16;

/// The schema of a B+Tree
#[derive(Clone, Eq, PartialEq)]
pub struct Schema {
    columns: Vec<Column>,
    block_size: usize,
    order: usize,
}

impl Schema {
    /// Construct a new B+Tree schema with the given `columns`.
    pub fn new(columns: Vec<Column>) -> TCResult<Self> {
        let mut key_size = 0;
        for col in &columns {
            if let Some(size) = col.dtype().size() {
                key_size += size;

                if col.max_len().is_some() {
                    return Err(bad_request!(
                        "maximum length is not applicable to a column of type {}",
                        col.dtype(),
                    ));
                }
            } else if let Some(size) = col.max_len() {
                key_size += size;
            } else {
                return Err(bad_request!(
                    "column of type {} requires a maximum length",
                    col.dtype(),
                ));
            }
        }

        fn index_size(order: usize, key_size: usize, data_size: usize) -> usize {
            let num_keys = data_size / key_size;
            let num_leaves = num_keys / order;

            let mut num_index_nodes = num_leaves / order;
            let mut index_size = num_leaves * (key_size + UUID_SIZE);
            while num_leaves > order {
                num_index_nodes /= order;
                index_size += num_index_nodes * (key_size + UUID_SIZE);
            }

            index_size
        }

        // calculate the minimum order such that 1TB of leaf data on disk needs 100MB of index data
        let mut order = MIN_ORDER;
        while index_size(order, key_size, TERA) > 100 * MEGA {
            order += 1;
        }

        Ok(Self {
            order,
            block_size: order * key_size,
            columns,
        })
    }

    /// Return an error if the given `range` does not match this [`Schema`].
    #[inline]
    pub fn validate_range(&self, range: Range) -> TCResult<Range> {
        if range.len() > self.columns.len() {
            return Err(bad_request!("{:?} has too many columns", range));
        }

        let (input_prefix, (start, end)) = range.into_inner();

        let mut prefix = Vec::with_capacity(input_prefix.len());
        for (value, column) in input_prefix.into_iter().zip(&self.columns) {
            let value = column.dtype.try_cast(value)?;
            prefix.push(value);
        }

        if start == Bound::Unbounded && end == Bound::Unbounded {
            Ok(Range::from_prefix(prefix))
        } else {
            let dtype = self.columns[prefix.len()].dtype;
            let validate_bound = |bound| match bound {
                Bound::Unbounded => Ok(Bound::Unbounded),
                Bound::Included(value) => dtype.try_cast(value).map(Bound::Included),
                Bound::Excluded(value) => dtype.try_cast(value).map(Bound::Excluded),
            };

            let start = validate_bound(start)?;
            let end = validate_bound(end)?;

            Ok(Range::with_bounds(prefix, (start, end)))
        }
    }
}

impl b_tree::Schema for Schema {
    type Error = TCError;
    type Value = Value;

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn len(&self) -> usize {
        self.columns.len()
    }

    fn order(&self) -> usize {
        self.order
    }

    fn validate(&self, key: Key) -> TCResult<Key> {
        if key.len() != self.len() {
            return Err(bad_request!(
                "BTree expected a key of length {}, not {}",
                self.len(),
                key.len()
            ));
        }

        key.into_iter()
            .zip(&self.columns)
            .map(|(val, col)| {
                val.into_type(col.dtype)
                    .ok_or_else(|| bad_request!("invalid value for column {}", &col.name))
            })
            .collect()
    }
}

impl<D: Digest> Hash<D> for Schema {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash(self.columns)
    }
}

impl<'a, D: Digest> Hash<D> for &'a Schema {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash(&self.columns)
    }
}

impl<'en> en::IntoStream<'en> for Schema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        self.columns.into_stream(encoder)
    }
}

impl<'en> en::ToStream<'en> for Schema {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        self.columns.to_stream(encoder)
    }
}

impl fmt::Debug for Schema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{:?}]", self.columns)
    }
}

/// A `Column` used in the schema of a [`BTree`].
#[derive(Clone, Eq, PartialEq)]
pub struct Column {
    pub name: Id,
    pub dtype: ValueType,
    pub max_len: Option<usize>,
}

impl Column {
    /// Get the name of this column.
    #[inline]
    pub fn name(&'_ self) -> &'_ Id {
        &self.name
    }

    /// Get the [`Class`] of this column.
    #[inline]
    pub fn dtype(&self) -> ValueType {
        self.dtype
    }

    /// Get the maximum size (in bytes) of this column.
    #[inline]
    pub fn max_len(&'_ self) -> &'_ Option<usize> {
        &self.max_len
    }
}

impl<D: Digest> Hash<D> for Column {
    fn hash(self) -> Output<D> {
        if let Some(max_len) = self.max_len {
            Hash::<D>::hash((self.name, self.dtype.path(), max_len))
        } else {
            Hash::<D>::hash((self.name, self.dtype.path()))
        }
    }
}

impl<'a, D: Digest> Hash<D> for &'a Column {
    fn hash(self) -> Output<D> {
        if let Some(max_len) = self.max_len {
            Hash::<D>::hash((&self.name, self.dtype.path(), max_len))
        } else {
            Hash::<D>::hash((&self.name, self.dtype.path()))
        }
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

impl CastFrom<Column> for Value {
    fn cast_from(column: Column) -> Self {
        Value::Tuple(
            vec![
                column.name.into(),
                column.dtype.path().into(),
                column.max_len.map(Value::from).into(),
            ]
            .into(),
        )
    }
}

struct ColumnVisitor;

#[async_trait]
impl de::Visitor for ColumnVisitor {
    type Value = Column;

    fn expecting() -> &'static str {
        "a Column definition"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let name = seq
            .next_element(())
            .await?
            .ok_or_else(|| de::Error::invalid_length(0, "a Column name"))?;

        let dtype = seq
            .next_element(())
            .await?
            .ok_or_else(|| de::Error::invalid_length(1, "a Column data type"))?;

        let max_len = seq.next_element(()).await?;

        Ok(Column {
            name,
            dtype,
            max_len,
        })
    }
}

#[async_trait]
impl de::FromStream for Column {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_seq(ColumnVisitor).await
    }
}

impl<'en> en::IntoStream<'en> for Column {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        if let Some(max_len) = self.max_len {
            (self.name, self.dtype, max_len).into_stream(encoder)
        } else {
            (self.name, self.dtype).into_stream(encoder)
        }
    }
}

impl<'en> en::ToStream<'en> for Column {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        if let Some(max_len) = self.max_len {
            en::IntoStream::into_stream((&self.name, &self.dtype, max_len), encoder)
        } else {
            en::IntoStream::into_stream((&self.name, &self.dtype), encoder)
        }
    }
}

impl<'a> From<&'a Column> for (&'a Id, ValueType) {
    fn from(col: &'a Column) -> (&'a Id, ValueType) {
        (&col.name, col.dtype)
    }
}

impl fmt::Debug for Column {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.max_len {
            Some(max_len) => write!(f, "{}: {}({})", self.name, self.dtype, max_len),
            None => write!(f, "{}: {}", self.name, self.dtype),
        }
    }
}
