use std::fmt;

use collate::Collate;

use super::{Value, ValueCollator};

#[derive(Clone, Eq, PartialEq)]
pub enum Bound {
    In(Value),
    Ex(Value),
    Unbounded,
}

impl fmt::Display for Bound {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::In(value) => write!(f, "include({})", value),
            Self::Ex(value) => write!(f, "exclude({})", value),
            Self::Unbounded => write!(f, "unbounded"),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct Range {
    pub start: Bound,
    pub end: Bound,
}

impl Range {
    pub fn contains_range(&self, inner: &Self, collator: &ValueCollator) -> bool {
        use std::cmp::Ordering::*;

        match &self.start {
            Bound::Unbounded => {}
            Bound::In(outer) => match &inner.start {
                Bound::Unbounded => return false,
                Bound::In(inner) => {
                    if collator.compare(inner, outer) == Less {
                        return false;
                    }
                }
                Bound::Ex(inner) => {
                    if collator.compare(inner, outer) != Greater {
                        return false;
                    }
                }
            },
            Bound::Ex(outer) => match &inner.start {
                Bound::Unbounded => return false,
                Bound::In(inner) => {
                    if collator.compare(inner, outer) != Greater {
                        return false;
                    }
                }
                Bound::Ex(inner) => {
                    if collator.compare(inner, outer) == Less {
                        return false;
                    }
                }
            },
        }

        match &self.end {
            Bound::Unbounded => {}
            Bound::In(outer) => match &inner.end {
                Bound::Unbounded => return false,
                Bound::In(inner) => {
                    if collator.compare(inner, outer) == Greater {
                        return false;
                    }
                }
                Bound::Ex(inner) => {
                    if collator.compare(inner, outer) != Less {
                        return false;
                    }
                }
            },
            Bound::Ex(outer) => match &inner.end {
                Bound::Unbounded => return false,
                Bound::In(inner) => {
                    if collator.compare(inner, outer) != Less {
                        return false;
                    }
                }
                Bound::Ex(inner) => {
                    if collator.compare(inner, outer) == Greater {
                        return false;
                    }
                }
            },
        }

        true
    }

    pub fn contains_value(&self, value: &Value, collator: &ValueCollator) -> bool {
        use std::cmp::Ordering::*;

        match &self.start {
            Bound::Unbounded => {}
            Bound::In(outer) => {
                if collator.compare(value, outer) == Less {
                    return false;
                }
            }
            Bound::Ex(outer) => {
                if collator.compare(value, outer) != Greater {
                    return false;
                }
            }
        }

        match &self.end {
            Bound::Unbounded => {}
            Bound::In(outer) => {
                if collator.compare(value, outer) == Greater {
                    return false;
                }
            }
            Bound::Ex(outer) => {
                if collator.compare(value, outer) != Less {
                    return false;
                }
            }
        }

        true
    }

    pub fn into_inner(self) -> (Bound, Bound) {
        (self.start, self.end)
    }
}

impl Default for Range {
    fn default() -> Self {
        Self {
            start: Bound::Unbounded,
            end: Bound::Unbounded,
        }
    }
}

impl fmt::Display for Range {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Range({}, {})", self.start, self.end)
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum Slice {
    Bound(Bound),
    Range(Range),
}

impl From<Range> for Slice {
    fn from(range: Range) -> Slice {
        Slice::Range(range)
    }
}

impl fmt::Display for Slice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bound(bound) => fmt::Display::fmt(bound, f),
            Self::Range(range) => fmt::Display::fmt(range, f),
        }
    }
}
