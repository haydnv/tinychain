use std::cmp::Ordering::{self, *};
use std::convert::TryInto;
use std::ops::Deref;

use crate::class::{Instance, TCResult};
use crate::error;
use crate::scalar::{Bound, Id, Number, StringType, TCString, Value, ValueType};

pub fn compare_value(left: &Value, right: &Value, dtype: ValueType) -> TCResult<Ordering> {
    left.expect(dtype, "for collation")?;
    right.expect(dtype, "for collation")?;
    match dtype {
        ValueType::Number(_) => {
            let left: &Number = left.try_into()?;
            let right: &Number = right.try_into()?;
            left.partial_cmp(right)
                .ok_or_else(|| error::unsupported("Unsupported Number comparison"))
        }
        ValueType::TCString(StringType::Id) => {
            let left: &Id = left.try_into()?;
            let right: &Id = right.try_into()?;
            left.as_str()
                .partial_cmp(right.as_str())
                .ok_or_else(|| error::unsupported("Unsupported String comparison"))
        }
        ValueType::TCString(StringType::UString) => {
            // TODO: support localized strings
            let left: &String = left.try_into()?;
            let right: &String = right.try_into()?;
            left.partial_cmp(right)
                .ok_or_else(|| error::unsupported("Unsupported String comparison"))
        }
        other => Err(error::bad_request("Collator does not support", other)),
    }
}

#[derive(Clone)]
pub struct Collator {
    schema: Vec<ValueType>,
}

impl Collator {
    pub fn new(schema: Vec<ValueType>) -> TCResult<Collator> {
        for dtype in &schema {
            if !Collator::supports(*dtype) {
                return Err(error::bad_request("Collation is not supported for", dtype));
            }
        }

        Ok(Collator { schema })
    }

    pub fn supports(dtype: ValueType) -> bool {
        use ValueType::*;
        match dtype {
            Number(_) => true,
            TCString(StringType::Id) => true,
            TCString(StringType::UString) => true,
            _ => false,
        }
    }

    pub fn is_sorted<V: Deref<Target = [Value]>>(&self, keys: &[V]) -> bool {
        for i in 1..keys.len() {
            if self.compare(&keys[i], &keys[i - 1]) == Less {
                return false;
            }
        }

        true
    }

    pub fn bisect_left<V: Deref<Target = [Value]>>(&self, keys: &[V], key: &[Value]) -> usize {
        if keys.is_empty() || key.is_empty() {
            return 0;
        }

        Self::_bisect_left(keys, |at| self.compare(at, key))
    }

    pub fn bisect_left_range<V: Deref<Target = [Value]>>(
        &self,
        keys: &[V],
        range: &[Bound],
    ) -> usize {
        if keys.is_empty() || range.is_empty() {
            return 0;
        }

        Self::_bisect_left(keys, |key| self.compare_bound(key, range, Less))
    }

    fn _bisect_left<'a, V: Deref<Target = [Value]>, F: Fn(&'a [Value]) -> Ordering>(
        keys: &'a [V],
        cmp: F,
    ) -> usize {
        let mut start = 0;
        let mut end = keys.len();

        while start < end {
            let mid = (start + end) / 2;

            if cmp(&keys[mid]) == Less {
                start = mid + 1;
            } else {
                end = mid;
            }
        }

        start
    }

    pub fn bisect_right<V: Deref<Target = [Value]>>(&self, keys: &[V], key: &[Value]) -> usize {
        if keys.is_empty() {
            return 0;
        }

        Self::_bisect_right(keys, |at| self.compare(at, key))
    }

    pub fn bisect_right_range<V: Deref<Target = [Value]>>(
        &self,
        keys: &[V],
        range: &[Bound],
    ) -> usize {
        if keys.is_empty() {
            0
        } else if range.is_empty() {
            keys.len()
        } else {
            Self::_bisect_right(keys, |key| self.compare_bound(key, range, Greater))
        }
    }

    fn _bisect_right<'a, V: Deref<Target = [Value]>, F: Fn(&'a [Value]) -> Ordering>(
        keys: &'a [V],
        cmp: F,
    ) -> usize {
        let mut start = 0;
        let mut end = keys.len();

        while start < end {
            let mid = (start + end) / 2;

            if cmp(&keys[mid]) == Greater {
                end = mid;
            } else {
                start = mid + 1;
            }
        }

        end
    }

    pub fn compare(&self, key1: &[Value], key2: &[Value]) -> Ordering {
        for i in 0..Ord::min(key1.len(), key2.len()) {
            match self.schema[i] {
                ValueType::Number(_) => {
                    let left: Number = key1[i].clone().try_into().unwrap();
                    let right: Number = key2[i].clone().try_into().unwrap();
                    if left < right {
                        return Less;
                    } else if left > right {
                        return Greater;
                    }
                }
                ValueType::TCString(st) => match st {
                    StringType::Id => {
                        let left: &Id = (&key1[i]).try_into().unwrap();
                        let right: &Id = (&key2[i]).try_into().unwrap();
                        match left.cmp(&right) {
                            Less => return Less,
                            Greater => return Greater,
                            _ => {}
                        }
                    }
                    StringType::UString => {
                        // TODO: add support for localized collation
                        let left: &String = (&key1[i]).try_into().unwrap();
                        let right: &String = (&key2[i]).try_into().unwrap();
                        match left.cmp(&right) {
                            Less => return Less,
                            Greater => return Greater,
                            _ => {}
                        }
                    }
                    _ => panic!("Collator::compare does not support {}", self.schema[i]),
                },
                _ => panic!("Collator::compare does not support {}", self.schema[i]),
            }
        }

        if key1.is_empty() && !key2.is_empty() {
            Less
        } else if !key1.is_empty() && key2.is_empty() {
            Greater
        } else {
            Equal
        }
    }

    pub fn compare_bound(
        &self,
        key: &[Value],
        range: &[Bound],
        exclude_order: Ordering,
    ) -> Ordering {
        use Bound::*;

        for i in 0..Ord::min(key.len(), range.len()) {
            match self.schema[i] {
                ValueType::Number(_) => match &range[i] {
                    Unbounded => {}
                    In(value) => match (&key[i], value) {
                        (Value::Number(left), Value::Number(right)) if left < right => return Less,
                        (Value::Number(left), Value::Number(right)) if left > right => {
                            return Greater
                        }
                        _ => {}
                    },
                    Ex(value) => {
                        return match (&key[i], value) {
                            (Value::Number(left), Value::Number(right)) if left < right => Less,
                            (Value::Number(left), Value::Number(right)) if left > right => Greater,
                            _ => exclude_order,
                        }
                    }
                },
                ValueType::TCString(StringType::UString) => match &range[i] {
                    Unbounded => {}
                    In(value) => match (&key[i], value) {
                        (
                            Value::TCString(TCString::UString(left)),
                            Value::TCString(TCString::UString(right)),
                        ) if left < right => return Less,
                        (
                            Value::TCString(TCString::UString(left)),
                            Value::TCString(TCString::UString(right)),
                        ) if left > right => return Greater,
                        _ => {}
                    },
                    Ex(value) => {
                        return match (&key[i], value) {
                            (
                                Value::TCString(TCString::UString(left)),
                                Value::TCString(TCString::UString(right)),
                            ) if left < right => Less,
                            (
                                Value::TCString(TCString::UString(left)),
                                Value::TCString(TCString::UString(right)),
                            ) if left > right => Greater,
                            _ => exclude_order,
                        }
                    }
                },
                _ => panic!("Collator::compare does not support {}", self.schema[i]),
            }
        }

        if key.is_empty() && range.iter().filter(|b| *b != &Bound::Unbounded).count() > 0 {
            exclude_order
        } else {
            Equal
        }
    }

    pub fn contains(&self, start: &[Bound], end: &[Bound], key: &[Value]) -> bool {
        self.compare_bound(key, start, Less) == Equal
            && self.compare_bound(key, end, Greater) == Equal
    }
}
