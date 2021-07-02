//! Provides common error types and associated convenience methods for Tinychain.
//!
//! This crate is a part of Tinychain: [http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain)

use std::fmt;

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
}

impl fmt::Debug for ErrorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for ErrorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BadGateway => f.write_str("bad gateway"),
            Self::BadRequest => f.write_str("bad request"),
            Self::Conflict => f.write_str("conflict"),
            Self::Forbidden => f.write_str("forbidden"),
            Self::Internal => f.write_str("internal error"),
            Self::MethodNotAllowed => f.write_str("method not allowed"),
            Self::NotFound => f.write_str("not found"),
            Self::NotImplemented => f.write_str("not implemented"),
            Self::Timeout => f.write_str("request timeout"),
            Self::Unauthorized => f.write_str("unauthorized"),
        }
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
    pub fn conflict() -> Self {
        Self {
            code: ErrorType::Conflict,
            message: String::default(),
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

#[cfg(feature = "tensor")]
impl From<afarray::ArrayError> for TCError {
    fn from(cause: afarray::ArrayError) -> Self {
        Self {
            code: ErrorType::Internal,
            message: format!("tensor error: {}", cause),
        }
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
