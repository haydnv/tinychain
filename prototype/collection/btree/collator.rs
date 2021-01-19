use std::cmp::Ordering::{self, *};
use std::convert::{TryFrom, TryInto};
use std::ops::Deref;

use crate::class::Instance;
use crate::error;
use crate::scalar::{Bound, Id, Number, NumberType, StringType, TCString, Value, ValueType};
use crate::TCResult;

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
            // don't support generic types (yet)
            Value => false,
            Number(NumberType::Number) => false,
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
            match self.compare_value(self.schema[i], &key1[i], &key2[i]) {
                Equal => {}
                rel => return rel,
            };
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

    pub fn compare_value(&self, dtype: ValueType, left: &Value, right: &Value) -> Ordering {
        assert_eq!(dtype, left.class());
        assert_eq!(dtype, right.class());

        match dtype {
            ValueType::Number(_) => {
                let left = Number::try_from(left).unwrap();
                let right = Number::try_from(right).unwrap();
                if left < right {
                    return Less;
                } else if left > right {
                    return Greater;
                }
            }
            ValueType::TCString(st) => match st {
                StringType::Id => {
                    let left: &Id = left.try_into().unwrap();
                    let right: &Id = right.try_into().unwrap();
                    match left.cmp(&right) {
                        Less => return Less,
                        Greater => return Greater,
                        _ => {}
                    }
                }
                StringType::UString => {
                    // TODO: add support for localized collation
                    let left: &String = left.try_into().unwrap();
                    let right: &String = right.try_into().unwrap();
                    match left.cmp(&right) {
                        Less => return Less,
                        Greater => return Greater,
                        _ => {}
                    }
                }
                _ => panic!("Collator::compare does not support {}", dtype),
            },
            _ => panic!("Collator::compare does not support {}", dtype),
        }

        Equal
    }

    pub fn contains(&self, start: &[Bound], end: &[Bound], key: &[Value]) -> bool {
        self.compare_bound(key, start, Less) == Equal
            && self.compare_bound(key, end, Greater) == Equal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::UIntType;

    #[test]
    fn test_compare() {
        let u64_type = ValueType::Number(NumberType::UInt(UIntType::U64));
        let collator = Collator {
            schema: vec![u64_type, u64_type],
        };
        let start = vec![Bound::In(0u64.into()), Bound::In(1u64.into())];
        let end = vec![Bound::Ex(2u64.into()), Bound::Ex(5u64.into())];
        let key = vec![0u64.into(), 5u64.into()];
        assert!(!collator.contains(&start, &end, &key));
    }
}
