use std::convert::TryInto;
use std::str::FromStr;

use rust_icu::loc::ULoc;

use crate::error;
use crate::value::TCResult;

pub struct Locale(ULoc);

impl FromStr for Locale {
    type Err = error::TCError;

    fn from_str(s: &str) -> TCResult<Locale> {
        let uloc: ULoc = s
            .try_into()
            .map_err(|e| error::bad_request(&format!("Unsupported locale code ({}): ", e), s))?;
        Ok(Locale(uloc))
    }
}
