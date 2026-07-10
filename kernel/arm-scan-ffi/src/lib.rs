//! C-ABI surface for the selective-scan kernel.
//!
//! Phase 1 exposes only a version probe so the Python loader can be wired
//! and tested early; the full scan entry point lands in Phase 4 behind this
//! same library. All raw-pointer handling is confined to this crate.

/// ABI version. The Python side checks this before calling anything else.
#[no_mangle]
pub extern "C" fn arm_scan_abi_version() -> u32 {
    1
}
