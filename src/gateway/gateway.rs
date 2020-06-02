use crate::kernel::Kernel;

use super::Hosted;

pub struct Gateway {
    kernel: Kernel,
    hosted: Hosted,
}
