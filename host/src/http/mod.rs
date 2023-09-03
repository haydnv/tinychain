//! The HTTP interface for `Gateway`.

use std::fmt;
use std::str::FromStr;

use hyper::header::HeaderValue;

use tc_error::*;

pub use client::*;
pub use server::*;

mod client;
mod server;

trait Accept: Default + FromStr {
    fn parse_header(header: Option<&HeaderValue>) -> TCResult<Self> {
        let header = if let Some(header) = header {
            header
                .to_str()
                .map_err(|cause| bad_request!("invalid Accept-Encoding header").consume(cause))?
        } else {
            return Ok(Self::default());
        };

        let accept = header.split(',');

        let mut quality = 0.;
        let mut encoding = None;
        for opt in accept {
            if opt.contains(';') {
                let opt: Vec<&str> = opt.split(';').collect();

                if opt.len() != 2 {
                    return Err(bad_request!(
                        "invalid encoding {} specified in Accept-Encoding header",
                        opt.join(";"),
                    ));
                }

                let format = opt[0].parse();
                let q = opt[1].parse().map_err(|e| {
                    bad_request!("invalid quality value {} in Accept-Encoding header", e)
                })?;

                if q > quality {
                    if let Ok(format) = format {
                        encoding = Some(format);
                        quality = q;
                    }
                }
            } else {
                if let Ok(format) = opt.parse() {
                    if encoding.is_none() {
                        encoding = Some(format);
                        quality = 1.;
                    }
                }
            }
        }

        Ok(encoding.unwrap_or_default())
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Encoding {
    Json,
    Tbon,
}

impl Encoding {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Json => "application/json",
            Self::Tbon => "application/tbon",
        }
    }
}

impl Default for Encoding {
    fn default() -> Self {
        Self::Json
    }
}

impl FromStr for Encoding {
    type Err = TCError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "application/json" => Ok(Self::Json),
            "application/tbon" => Ok(Self::Tbon),
            other => Err(bad_request!("encoding {} is not supported", other)),
        }
    }
}

impl Accept for Encoding {}

impl fmt::Display for Encoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
