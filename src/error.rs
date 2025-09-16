use core::fmt;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Message(&'static str),
    MessageOwned(String),
    Eof,
    InvalidHeader(u8),
    InvalidType(&'static str),
    InvalidSize,
    Unsupported(&'static str),
    Mismatch(&'static str),
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
        }
    }
}

impl std::error::Error for Error {}

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
