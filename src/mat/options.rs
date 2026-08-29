//! Conversion options for the `mat` feature, re-exported from `beve::mat`.
//!
//! These mirror the policy enums the underlying writer takes, so that no
//! hdf5-pure type reaches beve's public API; [`crate::mat`] has the rationale.
//! `MatV73Options::to_pure` is the single place the two vocabularies meet.
//!
//! A variant added upstream is not an error here and nothing flags it: the
//! mapping runs one way, so an unmirrored variant is simply unreachable.
//! Mirroring it is a step on the hdf5-pure upgrade checklist in `AGENTS.md`.
//! A variant *renamed or removed* upstream does not slip through — `to_pure`
//! stops compiling.

use hdf5_pure::mat as mat_pure;
use serde::{Deserialize, Serialize};

/// HDF5 dataset compression.
///
/// Anything other than [`None`](Self::None) requires
/// [`MatV73Options::libver`] to be [`LibVer::V110`] or newer, since compression
/// needs chunked storage and the chunk indices arrived in HDF5 1.10. The pair
/// is refused rather than resolved either way; see [`MatV73Options::libver`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Compression {
    /// No compression, and no chunking. The default, and the only setting
    /// MATLAB's `load` accepts under the default [`LibVer::V18`] format.
    None,
    /// HDF5 deflate compression.
    Deflate {
        /// zlib level, 0-9.
        level: u8,
        /// Apply the HDF5 byte-shuffle filter before deflating.
        shuffle: bool,
    },
}

/// Behavior when a BEVE object key is not a valid MATLAB identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum InvalidNamePolicy {
    /// Fail, naming the offending key.
    Error,
    /// Rewrite the key into a valid identifier, deduplicating collisions.
    Sanitize,
}

/// Behavior for a BEVE `null` appearing as an object value or an array element.
///
/// The file root is not one of those — it names no slot — so a root `null`
/// writes a MAT file with no variables under every policy except
/// [`Error`](Self::Error), which refuses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum NullPolicy {
    /// Write MATLAB `struct([])`, an empty struct array. The field stays
    /// present, so MATLAB code can reference it unconditionally.
    EmptyStructArray,
    /// Drop the field from its parent struct entirely, so `isfield` reports
    /// `false`.
    ///
    /// An array element is not droppable the way a field is: a cell array's
    /// element count is fixed by its dimensions, so a null element takes the
    /// `struct([])` marker regardless.
    Omit,
    /// Reject the `null` with an error, including at the root.
    Error,
}

/// Behavior for BEVE values with no direct MATLAB encoding: `bf16`, `f16`,
/// 128-bit integers, and unknown extensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum UnsupportedPolicy {
    /// Reject the value with an error naming its path.
    Error,
    /// Write the value's string representation as a MATLAB `string`. Applies to
    /// 128-bit integer scalars; low-precision floats want
    /// [`LossyNumericWidening`](Self::LossyNumericWidening) instead.
    StringFallback,
    /// Widen `bf16` and `f16` to MATLAB `single`.
    LossyNumericWidening,
}

/// Orientation for a one-dimensional BEVE array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum OneDimensionalMode {
    /// MATLAB shape `[N, 1]`.
    ColumnVector,
    /// MATLAB shape `[1, N]`.
    RowVector,
}

/// Behavior for a row-major matrix payload, i.e. BEVE `MatrixLayout::Right`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RowMajorPolicy {
    /// Transpose the payload into MATLAB's column-major layout.
    ReorderToColumnMajor,
    /// Fail rather than reorder.
    Error,
}

/// An HDF5 on-disk format boundary, mirroring HDF5's `H5F_libver_t`.
///
/// Variants are ordered oldest to newest and `Ord` follows that order, so
/// bounds compare directly. Only [`V18`](Self::V18) and [`V110`](Self::V110)
/// are reachable outputs: the writer produces nothing older than the HDF5 1.8
/// format, and nothing newer than 1.10 is needed by anything a MAT file
/// contains. The later boundaries are named so a bound can be stated in the
/// same vocabulary HDF5 uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum LibVer {
    /// The earliest format (HDF5 1.0+): version 0/1 superblock, v1 symbol-table
    /// groups. Readable by every released HDF5 library, but *not* writable —
    /// asking for it is refused rather than quietly satisfied with something
    /// newer.
    Earliest,
    /// HDF5 1.8: version 2 superblock, version 2 object headers, v2 B-tree
    /// indices. beve's default, and the newest format MATLAB's `load` accepts.
    V18,
    /// HDF5 1.10: version 3 superblock, plus the chunk indices [`Compression`]
    /// needs. MATLAB's MAT v7.3 loader refuses a file written this way.
    V110,
    /// HDF5 1.12.
    V112,
    /// HDF5 1.14.
    V114,
}

/// Options for BEVE -> MATLAB v7.3 conversion.
///
/// String values are always written as the modern MATLAB `string` class, which
/// is what real MATLAB's `save -v7.3` produces and what beve has always
/// emitted; it is not configurable.
///
/// A field absent from a serialized form takes its [`Default`], so a stored
/// configuration keeps loading after this struct gains a field. The cost is
/// that a truncated or hand-written blob no longer fails on the missing field;
/// it converts under the default instead.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
// On the struct rather than on each field. This type does gain fields --
// `libver` arrived in 6.0.0 and, without this, made every `MatV73Options`
// persisted under 5.x fail to load with "missing field `libver`". Per-field
// attributes would have to be remembered at every future addition, which is
// exactly the step that gets skipped.
#[serde(default)]
pub struct MatV73Options {
    pub compression: Compression,
    pub invalid_name_policy: InvalidNamePolicy,
    pub null_policy: NullPolicy,
    pub unsupported_policy: UnsupportedPolicy,
    pub one_dimensional_mode: OneDimensionalMode,
    pub row_major_policy: RowMajorPolicy,
    /// The newest HDF5 on-disk format the MAT file may use.
    ///
    /// Defaults to [`LibVer::V18`], the newest format every MATLAB release can
    /// open: a version 3 superblock is an HDF5 1.10 addition, and MATLAB's MAT
    /// v7.3 loader refuses one even on releases whose own libhdf5 reads it
    /// without difficulty.
    ///
    /// [`LibVer::V110`] is what [`Compression`] needs, since compression
    /// requires chunked storage and the chunk indices arrived in 1.10.
    /// Compression against a lower bound is refused rather than resolved either
    /// way, so raise this deliberately and accept that MATLAB will not `load`
    /// the result.
    pub libver: LibVer,
}

impl Default for MatV73Options {
    fn default() -> Self {
        Self {
            compression: Compression::None,
            invalid_name_policy: InvalidNamePolicy::Error,
            null_policy: NullPolicy::EmptyStructArray,
            unsupported_policy: UnsupportedPolicy::Error,
            one_dimensional_mode: OneDimensionalMode::ColumnVector,
            row_major_policy: RowMajorPolicy::ReorderToColumnMajor,
            libver: LibVer::V18,
        }
    }
}

// ---------------------------------------------------------------------------
// Mapping onto hdf5-pure
//
// Inherent private methods rather than `From` impls: a public
// `impl From<beve::mat::Compression> for hdf5_pure::mat::Compression` would put
// hdf5-pure back in beve's public API, which is the whole thing this module
// exists to avoid.
// ---------------------------------------------------------------------------

impl Compression {
    fn to_pure(self) -> mat_pure::Compression {
        match self {
            Compression::None => mat_pure::Compression::None,
            Compression::Deflate { level, shuffle } => {
                mat_pure::Compression::Deflate { level, shuffle }
            }
        }
    }
}

impl InvalidNamePolicy {
    fn to_pure(self) -> mat_pure::InvalidNamePolicy {
        match self {
            InvalidNamePolicy::Error => mat_pure::InvalidNamePolicy::Error,
            InvalidNamePolicy::Sanitize => mat_pure::InvalidNamePolicy::Sanitize,
        }
    }
}

impl NullPolicy {
    fn to_pure(self) -> mat_pure::NullPolicy {
        match self {
            NullPolicy::EmptyStructArray => mat_pure::NullPolicy::EmptyStructArray,
            NullPolicy::Omit => mat_pure::NullPolicy::Omit,
            NullPolicy::Error => mat_pure::NullPolicy::Error,
        }
    }
}

impl UnsupportedPolicy {
    fn to_pure(self) -> mat_pure::UnsupportedPolicy {
        match self {
            UnsupportedPolicy::Error => mat_pure::UnsupportedPolicy::Error,
            UnsupportedPolicy::StringFallback => mat_pure::UnsupportedPolicy::StringFallback,
            UnsupportedPolicy::LossyNumericWidening => {
                mat_pure::UnsupportedPolicy::LossyNumericWidening
            }
        }
    }
}

impl OneDimensionalMode {
    fn to_pure(self) -> mat_pure::OneDimensionalMode {
        match self {
            OneDimensionalMode::ColumnVector => mat_pure::OneDimensionalMode::ColumnVector,
            OneDimensionalMode::RowVector => mat_pure::OneDimensionalMode::RowVector,
        }
    }
}

impl RowMajorPolicy {
    fn to_pure(self) -> mat_pure::RowMajorPolicy {
        match self {
            RowMajorPolicy::ReorderToColumnMajor => mat_pure::RowMajorPolicy::ReorderToColumnMajor,
            RowMajorPolicy::Error => mat_pure::RowMajorPolicy::Error,
        }
    }
}

impl LibVer {
    fn to_pure(self) -> hdf5_pure::LibVer {
        match self {
            LibVer::Earliest => hdf5_pure::LibVer::Earliest,
            LibVer::V18 => hdf5_pure::LibVer::V18,
            LibVer::V110 => hdf5_pure::LibVer::V110,
            LibVer::V112 => hdf5_pure::LibVer::V112,
            LibVer::V114 => hdf5_pure::LibVer::V114,
        }
    }
}

impl MatV73Options {
    /// Project onto the underlying writer's options. Always pins
    /// `string_class = String`.
    ///
    /// `unit_variant_encoding` and `empty_sequence_policy` are deliberately not
    /// mirrored: the writer consults them only from its serde front end, to
    /// recover type information that serde withholds. BEVE carries that
    /// information in the document, so this walker reads the answer rather than
    /// choosing it. `empty_marker_encoding` is likewise left at the upstream
    /// default, which is the encoding beve has always written.
    pub(super) fn to_pure(&self) -> mat_pure::Options {
        let mut opts = mat_pure::Options::with_modern_strings();
        opts.compression = self.compression.to_pure();
        opts.invalid_name_policy = self.invalid_name_policy.to_pure();
        opts.null_policy = self.null_policy.to_pure();
        opts.unsupported_policy = self.unsupported_policy.to_pure();
        opts.one_dimensional_mode = self.one_dimensional_mode.to_pure();
        opts.row_major_policy = self.row_major_policy.to_pure();
        opts.libver = self.libver.to_pure();
        opts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every mirrored variant maps to the upstream variant of the same name.
    ///
    /// `to_pure` is a hand-written table, so a mistyped arm — `Omit` mapped to
    /// `Error` — compiles and converts silently. This is what catches that. It
    /// is also the only coverage for the variants no `.mat` fixture reaches:
    /// both `OneDimensionalMode`s, `UnsupportedPolicy::StringFallback`, and
    /// `LibVer::{Earliest, V112, V114}`.
    #[test]
    fn every_variant_maps_to_its_upstream_twin() {
        assert_eq!(Compression::None.to_pure(), mat_pure::Compression::None);
        assert_eq!(
            Compression::Deflate {
                level: 6,
                shuffle: true
            }
            .to_pure(),
            mat_pure::Compression::Deflate {
                level: 6,
                shuffle: true
            }
        );

        assert_eq!(
            InvalidNamePolicy::Error.to_pure(),
            mat_pure::InvalidNamePolicy::Error
        );
        assert_eq!(
            InvalidNamePolicy::Sanitize.to_pure(),
            mat_pure::InvalidNamePolicy::Sanitize
        );

        assert_eq!(
            NullPolicy::EmptyStructArray.to_pure(),
            mat_pure::NullPolicy::EmptyStructArray
        );
        assert_eq!(NullPolicy::Omit.to_pure(), mat_pure::NullPolicy::Omit);
        assert_eq!(NullPolicy::Error.to_pure(), mat_pure::NullPolicy::Error);

        assert_eq!(
            UnsupportedPolicy::Error.to_pure(),
            mat_pure::UnsupportedPolicy::Error
        );
        assert_eq!(
            UnsupportedPolicy::StringFallback.to_pure(),
            mat_pure::UnsupportedPolicy::StringFallback
        );
        assert_eq!(
            UnsupportedPolicy::LossyNumericWidening.to_pure(),
            mat_pure::UnsupportedPolicy::LossyNumericWidening
        );

        assert_eq!(
            OneDimensionalMode::ColumnVector.to_pure(),
            mat_pure::OneDimensionalMode::ColumnVector
        );
        assert_eq!(
            OneDimensionalMode::RowVector.to_pure(),
            mat_pure::OneDimensionalMode::RowVector
        );

        assert_eq!(
            RowMajorPolicy::ReorderToColumnMajor.to_pure(),
            mat_pure::RowMajorPolicy::ReorderToColumnMajor
        );
        assert_eq!(
            RowMajorPolicy::Error.to_pure(),
            mat_pure::RowMajorPolicy::Error
        );

        assert_eq!(LibVer::Earliest.to_pure(), hdf5_pure::LibVer::Earliest);
        assert_eq!(LibVer::V18.to_pure(), hdf5_pure::LibVer::V18);
        assert_eq!(LibVer::V110.to_pure(), hdf5_pure::LibVer::V110);
        assert_eq!(LibVer::V112.to_pure(), hdf5_pure::LibVer::V112);
        assert_eq!(LibVer::V114.to_pure(), hdf5_pure::LibVer::V114);

        // `LibVer` is ordered, so the mirror has to agree with upstream on the
        // order as well as the names: both ascending means beve's `Ord` sorts a
        // `libver` bound the way HDF5 does. Declared in one list so a variant
        // added out of order fails here rather than silently reordering.
        let ours = [
            LibVer::Earliest,
            LibVer::V18,
            LibVer::V110,
            LibVer::V112,
            LibVer::V114,
        ];
        assert!(ours.is_sorted(), "beve's LibVer is not oldest to newest");
        assert!(
            ours.map(LibVer::to_pure).is_sorted(),
            "beve's LibVer ordering disagrees with hdf5-pure's"
        );
    }

    /// A configuration stored before 6.0.0 added `libver` still loads, and the
    /// field it never carried arrives as the default. This is what `#[serde(default)]`
    /// on the struct buys; without it the blob failed with "missing field `libver`".
    #[test]
    fn an_options_blob_stored_before_libver_existed_still_loads() {
        let stored = r#"{"compression":"None","invalid_name_policy":"Error",
            "null_policy":"EmptyStructArray","unsupported_policy":"Error",
            "one_dimensional_mode":"ColumnVector",
            "row_major_policy":"ReorderToColumnMajor"}"#;

        let loaded: MatV73Options = serde_json::from_str(stored).unwrap();
        assert_eq!(loaded, MatV73Options::default());
        assert_eq!(loaded.libver, LibVer::V18);
    }

    /// The defaults beve documents, and the two upstream knobs it pins rather
    /// than exposes.
    #[test]
    fn defaults_pin_modern_strings_and_the_matlab_readable_format() {
        let opts = MatV73Options::default().to_pure();
        assert_eq!(opts.string_class, mat_pure::StringClass::String);
        assert_eq!(
            opts.empty_marker_encoding,
            mat_pure::EmptyMarkerEncoding::DataAsDims
        );
        assert_eq!(opts.libver, hdf5_pure::LibVer::V18);
        assert_eq!(opts.compression, mat_pure::Compression::None);
    }
}
