//! Provides common error types and associated convenience methods for TinyChain.
//!
//! This crate is a part of TinyChain: [http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain)

use std::convert::Infallible;
use std::fmt;

use destream::en;

/// A result of type `T`, or a [`TCError`]
pub type TCResult<T> = Result<T, TCError>;

struct ErrorData {
    message: String,
    stack: Vec<String>,
}

impl<'en> en::ToStream<'en> for ErrorData {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        if self.stack.is_empty() {
            return en::ToStream::to_stream(&self.message, encoder);
        }

        use en::EncodeMap;
        let mut map = encoder.encode_map(Some(2))?;
        map.encode_entry("message", &self.message)?;
        map.encode_entry("stack", &self.stack)?;
        map.end()
    }
}

impl<'en> en::IntoStream<'en> for ErrorData {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        if self.stack.is_empty() {
            return en::IntoStream::into_stream(self.message, encoder);
        }

        use en::EncodeMap;
        let mut map = encoder.encode_map(Some(2))?;
        map.encode_entry("message", self.message)?;
        map.encode_entry("stack", self.stack)?;
        map.end()
    }
}

impl<T> From<T> for ErrorData
where
    T: fmt::Display,
{
    fn from(message: T) -> Self {
        Self {
            message: message.to_string(),
            stack: vec![],
        }
    }
}

/// The category of a `TCError`.
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum ErrorKind {
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

impl<'en> en::IntoStream<'en> for ErrorKind {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
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

impl fmt::Debug for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for ErrorKind {
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
    kind: ErrorKind,
    data: ErrorData,
}

impl TCError {
    /// Returns a new error with the given code and message.
    pub fn new<I: fmt::Display>(code: ErrorKind, message: I) -> Self {
        Self {
            kind: code,
            data: message.into(),
        }
    }

    /// Reconstruct a [`TCError`] from its [`ErrorType`] and data.
    pub fn with_stack<I, S, SI>(code: ErrorKind, message: I, stack: S) -> Self
    where
        I: fmt::Display,
        SI: fmt::Display,
        S: IntoIterator<Item = SI>,
    {
        Self {
            kind: code,
            data: ErrorData {
                message: message.to_string(),
                stack: stack.into_iter().map(|msg| msg.to_string()).collect(),
            },
        }
    }

    /// Error indicating that the request is badly-constructed or nonsensical.
    pub fn bad_request<M: fmt::Display, I: fmt::Display>(message: M, cause: I) -> Self {
        let info = format!("{}: {}", message, cause);

        #[cfg(debug_assertions)]
        if info.starts_with("expected") {
            panic!("{}", info)
        }

        Self::new(ErrorKind::BadRequest, info)
    }

    /// Error indicating that the request depends on a resource which is exclusively locked
    /// by another request.
    pub fn conflict<M: fmt::Display>(message: M) -> Self {
        // #[cfg(debug_assertions)]
        // panic!("{}", message);
        //
        // #[cfg(not(debug_assertions))]
        Self::new(ErrorKind::Conflict, message)
    }

    /// Error indicating that the requested resource exists but does not support the request method.
    pub fn method_not_allowed<M: fmt::Display, S: fmt::Display, P: fmt::Display>(
        method: M,
        subject: S,
        path: P,
    ) -> Self {
        let message = format!("{} endpoint {} does not support {}", subject, path, method);

        #[cfg(debug_assertions)]
        panic!("{}", message);

        #[cfg(not(debug_assertions))]
        Self::new(ErrorKind::MethodNotAllowed, message)
    }

    /// Error indicating that the requested resource does not exist at the specified location.
    pub fn not_found<I: fmt::Display>(locator: I) -> Self {
        Self::new(ErrorKind::NotFound, locator)
    }

    /// Error indicating that a required feature is not yet implemented.
    pub fn not_implemented<F: fmt::Display>(feature: F) -> Self {
        Self::new(ErrorKind::NotImplemented, feature)
    }

    /// The [`ErrorKind`] of this error
    pub fn code(&self) -> ErrorKind {
        self.kind
    }

    /// The error message of this error
    pub fn message(&'_ self) -> &'_ str {
        &self.data.message
    }

    /// Construct a new error with the given `cause`
    pub fn consume<I: fmt::Display>(mut self, cause: I) -> Self {
        self.data.stack.push(cause.to_string());
        self
    }
}

impl std::error::Error for TCError {}

impl From<txn_lock::Error> for TCError {
    fn from(err: txn_lock::Error) -> Self {
        Self::conflict(err)
    }
}

impl From<Infallible> for TCError {
    fn from(_: Infallible) -> Self {
        unexpected!("an unanticipated error occurred--please file a bug report")
    }
}

impl<'en> en::ToStream<'en> for TCError {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        use en::EncodeMap;
        let mut map = encoder.encode_map(Some(1))?;
        map.encode_entry(self.kind, &self.data)?;
        map.end()
    }
}

impl<'en> en::IntoStream<'en> for TCError {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        use en::EncodeMap;
        let mut map = encoder.encode_map(Some(1))?;
        map.encode_entry(self.kind, self.data)?;
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
        write!(f, "{}: {}", self.kind, self.data.message)
    }
}


/// Error indicating that the an upstream server send an invalid response.
#[macro_export]
macro_rules! bad_gateway {
    ($($t:tt)*) => {{
        $crate::TCError::new($crate::ErrorKind::BadGateway, format!($($t)*))
    }}
}

/// Error indicating that the request is badly-constructed or nonsensical
#[macro_export]
macro_rules! bad_request {
    ($($t:tt)*) => {{
        $crate::TCError::new($crate::ErrorKind::BadRequest, format!($($t)*))
    }}
}

/// Error indicating that the requestor's credentials do not authorize the request to be fulfilled
#[macro_export]
macro_rules! forbidden {
    ($($t:tt)*) => {{
        $crate::TCError::new($crate::ErrorKind::Unavailable, format!($($t)*))
    }}
}

/// Error indicating that the request failed to complete in the allotted time.
#[macro_export]
macro_rules! timeout {
    ($($t:tt)*) => {{
        $crate::TCError::new($crate::ErrorKind::Timeout, format!($($t)*))
    }}
}

/// A truly unexpected error, for which no handling behavior can be defined
#[macro_export]
macro_rules! unexpected {
    ($($t:tt)*) => {{
        $crate::TCError::new($crate::ErrorKind::Internal, format!($($t)*))
    }}
}

/// Error indicating that the user's credentials are missing or nonsensical.
#[macro_export]
macro_rules! unauthorized {
    ($($t:tt)*) => {{
        $crate::TCError::new($crate::ErrorKind::Unauthorized, format!($($t)*))
    }}
}

/// Error indicating that this host is currently overloaded
#[macro_export]
macro_rules! unavailable {
    ($($t:tt)*) => {{
        $crate::TCError::new($crate::ErrorKind::Unavailable, format!($($t)*))
    }}
}
