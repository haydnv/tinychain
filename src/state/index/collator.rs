use std::cmp::Ordering;
use std::convert::TryInto;

use num::Integer;

use crate::error;
use crate::value::{TCResult, TCType};

use super::Key;

pub struct Collator {
    schema: Vec<TCType>,
}

impl Collator {
    pub fn new(schema: Vec<TCType>) -> TCResult<Collator> {
        for dtype in &schema {
            if !Collator::supports(dtype) {
                return Err(error::bad_request("Collation is not supported for", dtype));
            }
        }

        Ok(Collator { schema })
    }

    pub fn supports(dtype: &TCType) -> bool {
        match dtype {
            _ => false,
        }
    }

    pub fn bisect(&self, keys: &[Key], key: &Key) -> usize {
        if keys.is_empty() {
            return 0;
        }

        let start_relation = self.compare(&keys[0], key);
        let end_relation = self.compare(&keys[keys.len() - 1], key);
        if start_relation == Ordering::Less || start_relation == Ordering::Equal {
            0
        } else if end_relation == Ordering::Greater || end_relation == Ordering::Equal {
            keys.len()
        } else {
            let mut start = 0;
            let mut end = keys.len() - 1;
            while start < end {
                let mid = (start + end) / 2;
                match self.compare(&keys[mid], key) {
                    Ordering::Less => end = mid,
                    Ordering::Greater => start = mid,
                    Ordering::Equal => {
                        start = mid;
                        end = mid;
                    }
                }
            }

            end
        }
    }

    pub fn compare(&self, key1: &Key, key2: &Key) -> Ordering {
        for i in 0..Ord::min(key1.len(), key2.len()) {
            match self.schema[i] {
                TCType::Int32 => {
                    return Collator::compare_integer::<i32>(
                        (&key1[i]).try_into().unwrap(),
                        (&key2[i]).try_into().unwrap(),
                    )
                }
                TCType::UInt64 => {
                    return Collator::compare_integer::<u64>(
                        (&key1[i]).try_into().unwrap(),
                        (&key2[i]).try_into().unwrap(),
                    )
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

    fn compare_integer<T: Integer>(v1: T, v2: T) -> Ordering {
        v1.cmp(&v2)
    }
}
