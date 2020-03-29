pub enum Code {
    MethodNotAllowed,
}

pub struct TCError {
    reason: Code,
    message: String,
}

impl TCError {
    pub fn of(reason: Code, message: String) -> TCError {
        TCError { reason, message }
    }
}

pub fn method_not_allowed() -> TCError {
    TCError::of(
        Code::MethodNotAllowed,
        "This resource does not support this request method".to_string(),
    )
}
