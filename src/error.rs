use core::fmt;

pub type Result<T> = core::result::Result<T, Error>;

/// Marked `#[non_exhaustive]` so that naming a new failure mode is not itself a
/// breaking change: adding `RecursionLimitExceeded` broke every downstream
/// exhaustive `match`, and a decoder will keep learning to refuse things.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    Message(&'static str),
    MessageOwned(String),
    Eof,
    InvalidHeader(u8),
    InvalidType(&'static str),
    InvalidSize,
    Unsupported(&'static str),
    Mismatch(&'static str),
    /// Input nested deeper than [`MAX_RECURSION_DEPTH`](crate::MAX_RECURSION_DEPTH).
    ///
    /// Decoding a nested value recurses one native stack frame per level, and a
    /// Rust stack overflow aborts the process instead of unwinding, so no caller
    /// can catch it. Refusing the input is the only outcome a caller can act on.
    RecursionLimitExceeded,
}

impl Error {
    pub fn msg<T: Into<String>>(msg: T) -> Self {
        Self::MessageOwned(msg.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Message(m) => write!(f, "{}", m),
            Error::MessageOwned(m) => write!(f, "{}", m),
            Error::Eof => write!(f, "unexpected end of input"),
            Error::InvalidHeader(h) => write!(f, "invalid header: 0x{h:02x}"),
            Error::InvalidType(t) => write!(f, "invalid type: {t}"),
            Error::InvalidSize => write!(f, "invalid size"),
            Error::Unsupported(s) => write!(f, "unsupported: {s}"),
            Error::Mismatch(s) => write!(f, "type mismatch: {s}"),
            Error::RecursionLimitExceeded => write!(
                f,
                "input nests deeper than the maximum of {} levels",
                crate::MAX_RECURSION_DEPTH
            ),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::MessageOwned(e.to_string())
    }
}

impl serde::ser::Error for Error {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        Error::MessageOwned(msg.to_string())
    }
}

impl serde::de::Error for Error {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        Error::MessageOwned(msg.to_string())
    }
}
