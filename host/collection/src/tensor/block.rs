use std::fmt;

use ha_ndarray::*;

use tc_error::*;
use tc_value::{Float, Int, Number, UInt};

pub enum Block {
    F32(Array<f32>),
    F64(Array<f64>),
    I16(Array<i16>),
    I32(Array<i32>),
    I64(Array<i64>),
    U8(Array<u8>),
    U16(Array<u16>),
    U32(Array<u32>),
    U64(Array<u64>),
}

macro_rules! block_dispatch {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            Block::F32($var) => $call,
            Block::F64($var) => $call,
            Block::I16($var) => $call,
            Block::I32($var) => $call,
            Block::I64($var) => $call,
            Block::U8($var) => $call,
            Block::U16($var) => $call,
            Block::U32($var) => $call,
            Block::U64($var) => $call,
        }
    };
}

macro_rules! block_cmp {
    ($self:ident, $other:ident, $this:ident, $that:ident, $call:expr) => {
        match ($self, $other) {
            (Self::F32($this), Self::F32($that)) => {
                $this.eq($that).map(Array::from).map_err(TCError::from)
            }
            (Self::F64($this), Self::F64($that)) => {
                $this.eq($that).map(Array::from).map_err(TCError::from)
            }
            (Self::I16($this), Self::I16($that)) => {
                $this.eq($that).map(Array::from).map_err(TCError::from)
            }
            (Self::I32($this), Self::I32($that)) => {
                $this.eq($that).map(Array::from).map_err(TCError::from)
            }
            (Self::I64($this), Self::I64($that)) => {
                $this.eq($that).map(Array::from).map_err(TCError::from)
            }
            (Self::U8($this), Self::U8($that)) => {
                $this.eq($that).map(Array::from).map_err(TCError::from)
            }
            (Self::U16($this), Self::U16($that)) => {
                $this.eq($that).map(Array::from).map_err(TCError::from)
            }
            (Self::U32($this), Self::U32($that)) => {
                $this.eq($that).map(Array::from).map_err(TCError::from)
            }
            (Self::U64($this), Self::U64($that)) => {
                $this.eq($that).map(Array::from).map_err(TCError::from)
            }
            ($this, $that) => Err(bad_request!("cannot compare {:?} with {:?}", $this, $that)),
        }
    };
}

macro_rules! block_cmp_scalar {
    ($self:ident, $other:ident, $this:ident, $that:ident, $call:expr) => {
        match ($self, $other) {
            (Self::F32($this), Number::Float(Float::F32($that))) => $this
                .eq_scalar($that)
                .map(Array::from)
                .map_err(TCError::from),

            (Self::F64($this), Number::Float(Float::F64($that))) => $this
                .eq_scalar($that)
                .map(Array::from)
                .map_err(TCError::from),

            (Self::I16($this), Number::Int(Int::I16($that))) => $this
                .eq_scalar($that)
                .map(Array::from)
                .map_err(TCError::from),

            (Self::I32($this), Number::Int(Int::I32($that))) => $this
                .eq_scalar($that)
                .map(Array::from)
                .map_err(TCError::from),

            (Self::I64($this), Number::Int(Int::I64($that))) => $this
                .eq_scalar($that)
                .map(Array::from)
                .map_err(TCError::from),

            (Self::U8($this), Number::UInt(UInt::U8($that))) => $this
                .eq_scalar($that)
                .map(Array::from)
                .map_err(TCError::from),

            (Self::U16($this), Number::UInt(UInt::U16($that))) => $this
                .eq_scalar($that)
                .map(Array::from)
                .map_err(TCError::from),

            (Self::U32($this), Number::UInt(UInt::U32($that))) => $this
                .eq_scalar($that)
                .map(Array::from)
                .map_err(TCError::from),

            (Self::U64($this), Number::UInt(UInt::U64($that))) => $this
                .eq_scalar($that)
                .map(Array::from)
                .map_err(TCError::from),

            ($this, $that) => Err(bad_request!("cannot compare {:?} with {:?}", $this, $that)),
        }
    };
}

impl Block {
    pub fn cast<T: CDatatype>(self) -> TCResult<Array<T>> {
        block_dispatch!(
            self,
            this,
            this.cast().map(Array::from).map_err(TCError::from)
        )
    }

    pub fn and(self, other: Self) -> TCResult<Array<u8>> {
        block_cmp!(
            self,
            other,
            this,
            that,
            this.and(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn and_scalar(self, other: Number) -> TCResult<Array<u8>> {
        block_cmp_scalar!(
            self,
            other,
            this,
            that,
            this.and_scalar(that)
                .map(Array::from)
                .map_err(TCError::from)
        )
    }

    pub fn not(self) -> TCResult<Array<u8>> {
        block_dispatch!(
            self,
            this,
            this.not().map(Array::from).map_err(TCError::from)
        )
    }

    pub fn or(self, other: Self) -> TCResult<Array<u8>> {
        block_cmp!(
            self,
            other,
            this,
            that,
            this.or(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn or_scalar(self, other: Number) -> TCResult<Array<u8>> {
        block_cmp_scalar!(
            self,
            other,
            this,
            that,
            this.or_scalar(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn xor(self, other: Self) -> TCResult<Array<u8>> {
        block_cmp!(
            self,
            other,
            this,
            that,
            this.xor(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn xor_scalar(self, other: Number) -> TCResult<Array<u8>> {
        block_cmp_scalar!(
            self,
            other,
            this,
            that,
            this.xor_scalar(that)
                .map(Array::from)
                .map_err(TCError::from)
        )
    }

    pub fn eq(self, other: Self) -> TCResult<Array<u8>> {
        block_cmp!(
            self,
            other,
            this,
            that,
            this.eq(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn eq_scalar(self, other: Number) -> TCResult<Array<u8>> {
        block_cmp_scalar!(
            self,
            other,
            this,
            that,
            this.eq_scalar(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn gt(self, other: Self) -> TCResult<Array<u8>> {
        block_cmp!(
            self,
            other,
            this,
            that,
            this.gt(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn gt_scalar(self, other: Number) -> TCResult<Array<u8>> {
        block_cmp_scalar!(
            self,
            other,
            this,
            that,
            this.gt_scalar(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn ge(self, other: Self) -> TCResult<Array<u8>> {
        block_cmp!(
            self,
            other,
            this,
            that,
            this.ge(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn ge_scalar(self, other: Number) -> TCResult<Array<u8>> {
        block_cmp_scalar!(
            self,
            other,
            this,
            that,
            this.ge_scalar(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn lt(self, other: Self) -> TCResult<Array<u8>> {
        block_cmp!(
            self,
            other,
            this,
            that,
            this.lt(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn lt_scalar(self, other: Number) -> TCResult<Array<u8>> {
        block_cmp_scalar!(
            self,
            other,
            this,
            that,
            this.lt_scalar(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn le(self, other: Self) -> TCResult<Array<u8>> {
        block_cmp!(
            self,
            other,
            this,
            that,
            this.le(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn le_scalar(self, other: Number) -> TCResult<Array<u8>> {
        block_cmp_scalar!(
            self,
            other,
            this,
            that,
            this.le_scalar(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn ne(self, other: Self) -> TCResult<Array<u8>> {
        block_cmp!(
            self,
            other,
            this,
            that,
            this.ne(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn ne_scalar(self, other: Number) -> TCResult<Array<u8>> {
        block_cmp_scalar!(
            self,
            other,
            this,
            that,
            this.ne_scalar(that).map(Array::from).map_err(TCError::from)
        )
    }
}

macro_rules! block_from {
    ($t:ty, $var:ident) => {
        impl From<Array<$t>> for Block {
            fn from(array: Array<$t>) -> Self {
                Self::$var(array)
            }
        }
    };
}

block_from!(f32, F32);
block_from!(f64, F64);
block_from!(i16, I16);
block_from!(i32, I32);
block_from!(i64, I64);
block_from!(u8, U8);
block_from!(u16, U16);
block_from!(u32, U32);
block_from!(u64, U64);

impl fmt::Debug for Block {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::F32(this) => this.fmt(f),
            Self::F64(this) => this.fmt(f),
            Self::I16(this) => this.fmt(f),
            Self::I32(this) => this.fmt(f),
            Self::I64(this) => this.fmt(f),
            Self::U8(this) => this.fmt(f),
            Self::U16(this) => this.fmt(f),
            Self::U32(this) => this.fmt(f),
            Self::U64(this) => this.fmt(f),
        }
    }
}
