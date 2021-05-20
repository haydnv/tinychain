use afarray::Array;
use number_general::NumberType;

use tc_transact::fs::File;

use crate::Shape;

#[derive(Clone)]
pub struct BlockListFile<F: File<Array>> {
    file: F,
    dtype: NumberType,
    shape: Shape,
}
