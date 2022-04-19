//! Provides common error types and associated convenience methods for TinyChain.
//!
//! This crate is a part of TinyChain: [http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain)

use std::fmt;

use destream::{en, EncodeMap, Encoder};

pub type TCResult<T> = Result<T, TCError>;

/// The category of a `TCError`.
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum ErrorType {
    BadGateway,
    BadRequest,
    Conflict,
    Forbidden,
    Internal,
    MethodNotAllowed,
    NotFound,
    NotImplemented,
    Timeout,
    Unauthorized,
    Unavailable,
}

impl<'en> en::IntoStream<'en> for ErrorType {
    fn into_stream<E: Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        format!(
            "/error/{}",
            match self {
                Self::BadGateway => "bad_gateway",
                Self::BadRequest => "bad_request",
                Self::Conflict => "conflict",
                Self::Forbidden => "forbidden",
                Self::Internal => "internal",
                Self::MethodNotAllowed => "method_not_allowed",
                Self::NotFound => "not_found",
                Self::NotImplemented => "not_implemented",
                Self::Timeout => "timeout",
                Self::Unauthorized => "unauthorized",
                Self::Unavailable => "temporarily unavailable",
            }
        )
        .into_stream(encoder)
    }
}

impl fmt::Debug for ErrorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for ErrorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Self::BadGateway => "bad gateway",
            Self::BadRequest => "bad request",
            Self::Conflict => "conflict",
            Self::Forbidden => "forbidden",
            Self::Internal => "internal error",
            Self::MethodNotAllowed => "method not allowed",
            Self::NotFound => "not found",
            Self::NotImplemented => "not implemented",
            Self::Timeout => "request timeout",
            Self::Unauthorized => "unauthorized",
            Self::Unavailable => "temporarily unavailable",
        })
    }
}

/// A general error description.
pub struct TCError {
    code: ErrorType,
    message: String,
}

impl TCError {
    /// Returns a new error with the given code and message.
    pub fn new(code: ErrorType, message: String) -> Self {
        Self { code, message }
    }

    /// Error indicating that the an upstream server send an invalid response.
    pub fn bad_gateway<I: fmt::Display>(cause: I) -> Self {
        Self {
            code: ErrorType::BadGateway,
            message: cause.to_string(),
        }
    }

    /// Error indicating that the request is badly-constructed or nonsensical.
    pub fn bad_request<M: fmt::Display, I: fmt::Display>(message: M, cause: I) -> Self {
        Self {
            code: ErrorType::BadRequest,
            message: format!("{}: {}", message, cause),
        }
    }

    /// Error indicating that the request depends on a resource which is exclusively locked
    /// by another request.
    pub fn conflict<M: fmt::Display>(message: M) -> Self {
        Self {
            code: ErrorType::Conflict,
            message: message.to_string(),
        }
    }

    /// Error indicating that the request actor's credentials do not authorize access to some
    /// request dependencies.
    pub fn forbidden<M: fmt::Display, I: fmt::Display>(message: M, id: I) -> Self {
        Self {
            code: ErrorType::Forbidden,
            message: format!("{}: {}", message, id),
        }
    }

    /// A truly unexpected error, for which the calling application cannot define any specific
    /// handling behavior.
    pub fn internal<I: fmt::Display>(info: I) -> Self {
        #[cfg(debug_assertions)]
        panic!("{}", info);

        #[cfg(not(debug_assertions))]
        Self {
            code: ErrorType::Internal,
            message: info.to_string(),
        }
    }

    /// Error indicating that the requested resource exists but does not support the request method.
    pub fn method_not_allowed<M: fmt::Display, S: fmt::Display, P: fmt::Display>(
        method: M,
        subject: S,
        path: P,
    ) -> Self {
        Self {
            code: ErrorType::MethodNotAllowed,
            message: format!("{} endpoint {} does not support {}", subject, path, method),
        }
    }

    /// Error indicating that the requested resource does not exist at the specified location.
    pub fn not_found<I: fmt::Display>(locator: I) -> Self {
        Self {
            code: ErrorType::NotFound,
            message: locator.to_string(),
        }
    }

    /// Error indicating that a required feature is not yet implemented.
    pub fn not_implemented<F: fmt::Display>(feature: F) -> Self {
        Self {
            code: ErrorType::NotImplemented,
            message: feature.to_string(),
        }
    }

    /// Error indicating that the request failed to complete in the allotted time.
    pub fn timeout<I: fmt::Display>(info: I) -> Self {
        Self {
            code: ErrorType::Timeout,
            message: info.to_string(),
        }
    }

    /// Error indicating that the user's credentials are missing or nonsensical.
    pub fn unauthorized<I: fmt::Display>(info: I) -> Self {
        Self {
            code: ErrorType::Unauthorized,
            message: format!("invalid credentials: {}", info),
        }
    }

    /// Error indicating that this host is currently overloaded
    pub fn unavailable<I: fmt::Display>(info: I) -> Self {
        Self {
            code: ErrorType::Unavailable,
            message: info.to_string(),
        }
    }

    /// Error indicating that the request is badly-constructed or nonsensical.
    pub fn unsupported<I: fmt::Display>(info: I) -> Self {
        Self {
            code: ErrorType::BadRequest,
            message: info.to_string(),
        }
    }

    pub fn code(&self) -> ErrorType {
        self.code
    }

    pub fn message(&'_ self) -> &'_ str {
        &self.message
    }

    pub fn consume<I: fmt::Display>(self, info: I) -> Self {
        Self {
            code: self.code,
            message: format!("{}: {}", info, self.message),
        }
    }
}

impl std::error::Error for TCError {}

impl<'en> en::ToStream<'en> for TCError {
    fn to_stream<E: Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        let mut map = encoder.encode_map(Some(1))?;
        map.encode_entry(self.code, &self.message)?;
        map.end()
    }
}

impl<'en> en::IntoStream<'en> for TCError {
    fn into_stream<E: Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let mut map = encoder.encode_map(Some(1))?;
        map.encode_entry(self.code, self.message)?;
        map.end()
    }
}

impl fmt::Debug for TCError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for TCError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}
