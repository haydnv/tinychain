use std::cmp::Ordering::{self, *};
use std::convert::TryInto;

use crate::error;
use crate::value::{Number, Value, TCResult};
use crate::value::class::ValueType;

pub struct Collator {
    schema: Vec<ValueType>,
}

impl Collator {
    pub fn new(schema: Vec<ValueType>) -> TCResult<Collator> {
        for dtype in &schema {
            if !Collator::supports(dtype) {
                return Err(error::bad_request("Collation is not supported for", dtype));
            }
        }

        Ok(Collator { schema })
    }

    pub fn supports(dtype: &ValueType) -> bool {
        use ValueType::*;
        match dtype {
            Number(_) => true,
            _ => false,
        }
    }

    pub fn bisect(&self, keys: &[&[Value]], key: &[Value]) -> usize {
        if keys.is_empty() {
            return 0;
        }

        let start_relation = self.compare(&keys[0], key);
        let end_relation = self.compare(&keys[keys.len() - 1], key);
        if start_relation == Less {
            0
        } else if end_relation == Greater || end_relation == Equal {
            keys.len()
        } else {
            let mut start = 0;
            let mut end = keys.len() - 1;
            while start < end {
                let mid = (start + end) / 2;
                match self.compare(&keys[mid], key) {
                    Less => start = mid,
                    Greater => end = mid,
                    Equal if mid == keys.len() - 1 => end = mid,
                    Equal => match self.compare(&keys[mid + 1], key) {
                        Greater => end = mid,
                        Equal => start = mid + 1,
                        Less => panic!("Tried to collate a non-sorted Vec!"),
                    },
                }
            }

            end
        }
    }

    pub fn bisect_left(&self, keys: &[&[Value]], key: &[Value]) -> usize {
        if keys.is_empty() {
            return 0;
        }

        let start_relation = self.compare(&keys[0], key);
        let end_relation = self.compare(&keys[keys.len() - 1], key);
        if start_relation == Greater || start_relation == Equal {
            0
        } else if end_relation == Less {
            keys.len()
        } else {
            let mut start = 0;
            let mut end = keys.len() - 1;
            while start < end {
                let mid = (start + end) / 2;
                match self.compare(&keys[mid], key) {
                    Less => start = mid,
                    Greater => end = mid,
                    Equal if mid == 0 => return 0,
                    Equal => match self.compare(&keys[mid - 1], key) {
                        Equal => end = mid - 1,
                        Less => start = mid,
                        Greater => panic!("Tried to collate a non-sorted Vec!"),
                    },
                }
            }

            start
        }
    }

    pub fn compare(&self, key1: &[Value], key2: &[Value]) -> Ordering {
        for i in 0..Ord::min(key1.len(), key2.len()) {
            match self.schema[i] {
                ValueType::Number(_) => {
                    let left: Number = key1[i].clone().try_into().unwrap();
                    let right: Number = key2[i].clone().try_into().unwrap();
                    if left < right {
                        return Ordering::Less;
                    } else if left > right {
                        return Ordering::Greater;
                    }
                }
                _ => panic!("Collator::compare does not support {}", self.schema[i]),
            }
        }

        if key1.is_empty() && !key2.is_empty() {
            Ordering::Less
        } else if !key1.is_empty() && key2.is_empty() {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}
