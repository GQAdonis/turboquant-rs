use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum TurboQuantError {
    #[error("dimension {0} is not a power of two")]
    DimensionNotPowerOfTwo(usize),

    #[error("bit width {bits} is unsupported; must be 2, 3, or 4")]
    UnsupportedBitWidth { bits: u8 },

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("input is empty")]
    EmptyInput,
}

pub type Result<T> = std::result::Result<T, TurboQuantError>;
