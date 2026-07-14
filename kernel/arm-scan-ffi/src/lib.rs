//! C-ABI surface for the selective-scan kernel.
//!
//! One entry point, contiguous row-major f32 tensors only (the Python
//! wrapper calls `.contiguous()`). All raw-pointer handling in the project
//! lives in this crate; a Rust panic is caught at the boundary and
//! reported as an error code instead of unwinding into the caller.
//!
//! Layout contract (identical to arm-scan-core's module docs):
//!   u, delta, z, out : (batch, dim, len)
//!   a                : (dim, state)
//!   b, c             : (batch, groups, state, len)  — pass groups=1 for
//!                      the standard (B, N, L) case, same memory layout
//!   d_skip, delta_bias : (dim,)
//!   last_state       : (batch, dim, state)
//! Nullable: d_skip, z, delta_bias, last_state. Everything else non-null.

use std::os::raw::c_int;

use arm_scan_core::{
    selective_scan_with_options, Backend, ScanDims, ScanError, ScanInput, ScanOptions, Threading,
};

/// ABI version. The Python loader checks this before calling anything else.
/// Bump on any signature or semantic change to `arm_scan_selective_scan_f32`.
///
/// 3: added the `reverse` parameter (backward-in-time traversal).
#[no_mangle]
pub extern "C" fn arm_scan_abi_version() -> u32 {
    3
}

/// Dimensions for a scan call. `groups` must divide `dim`.
#[repr(C)]
pub struct ArmScanDims {
    pub batch: usize,
    pub dim: usize,
    pub len: usize,
    pub state: usize,
    pub groups: usize,
}

// Return codes for arm_scan_selective_scan_f32.
pub const ARM_SCAN_OK: c_int = 0;
pub const ARM_SCAN_ERR_NULL_POINTER: c_int = 1;
pub const ARM_SCAN_ERR_INVALID_DIMS: c_int = 2;
pub const ARM_SCAN_ERR_BACKEND_UNAVAILABLE: c_int = 3;
pub const ARM_SCAN_ERR_BAD_ENUM: c_int = 4;
pub const ARM_SCAN_ERR_PANIC: c_int = 5;

fn backend_from(v: c_int) -> Option<Backend> {
    match v {
        0 => Some(Backend::Auto),
        1 => Some(Backend::Scalar),
        2 => Some(Backend::Neon),
        _ => None,
    }
}

fn threading_from(v: c_int) -> Option<Threading> {
    match v {
        0 => Some(Threading::Auto),
        1 => Some(Threading::Sequential),
        2 => Some(Threading::Rayon),
        _ => None,
    }
}

/// Run the selective scan.
///
/// `backend`: 0 = auto, 1 = scalar, 2 = neon.
/// `threading`: 0 = auto, 1 = sequential, 2 = rayon.
/// `delta_softplus`: nonzero to apply softplus(delta + delta_bias) inside
/// the kernel.
/// `reverse`: nonzero to walk the sequence backward in time. Output layout is
/// unchanged (timestep `t` still lands at index `t`); only the recurrence's
/// traversal order flips. Equivalent to flipping the time axis of u/delta/b/c/z,
/// scanning forward, and flipping the output back — without the copies.
///
/// Returns `ARM_SCAN_OK` (0) on success, a nonzero code otherwise; `out`
/// contents are unspecified on error.
///
/// # Safety
/// Every non-null pointer must reference a readable (writable for `out`,
/// `last_state`) buffer of exactly the element count implied by `dims` and
/// the layout contract in the module docs, valid for the duration of the
/// call. Buffers must not overlap.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn arm_scan_selective_scan_f32(
    dims: *const ArmScanDims,
    u: *const f32,
    delta: *const f32,
    a: *const f32,
    b: *const f32,
    c: *const f32,
    d_skip: *const f32,
    z: *const f32,
    delta_bias: *const f32,
    delta_softplus: c_int,
    reverse: c_int,
    backend: c_int,
    threading: c_int,
    out: *mut f32,
    last_state: *mut f32,
) -> c_int {
    if dims.is_null()
        || u.is_null()
        || delta.is_null()
        || a.is_null()
        || b.is_null()
        || c.is_null()
        || out.is_null()
    {
        return ARM_SCAN_ERR_NULL_POINTER;
    }
    let (Some(backend), Some(threading)) = (backend_from(backend), threading_from(threading))
    else {
        return ARM_SCAN_ERR_BAD_ENUM;
    };

    let d = &*dims;
    // Overflow-checked element counts before any slice is formed.
    let Some(bdl) = d
        .batch
        .checked_mul(d.dim)
        .and_then(|v| v.checked_mul(d.len))
    else {
        return ARM_SCAN_ERR_INVALID_DIMS;
    };
    let Some(bgnl) = d
        .batch
        .checked_mul(d.groups)
        .and_then(|v| v.checked_mul(d.state))
        .and_then(|v| v.checked_mul(d.len))
    else {
        return ARM_SCAN_ERR_INVALID_DIMS;
    };
    let Some(dn) = d.dim.checked_mul(d.state) else {
        return ARM_SCAN_ERR_INVALID_DIMS;
    };
    let Some(bdn) = d
        .batch
        .checked_mul(d.dim)
        .and_then(|v| v.checked_mul(d.state))
    else {
        return ARM_SCAN_ERR_INVALID_DIMS;
    };

    let scan_dims = ScanDims {
        batch: d.batch,
        dim: d.dim,
        len: d.len,
        state: d.state,
        groups: d.groups,
    };

    let opt = |p: *const f32, n: usize| {
        if p.is_null() {
            None
        } else {
            Some(std::slice::from_raw_parts(p, n))
        }
    };
    let input = ScanInput {
        u: std::slice::from_raw_parts(u, bdl),
        delta: std::slice::from_raw_parts(delta, bdl),
        a: std::slice::from_raw_parts(a, dn),
        b: std::slice::from_raw_parts(b, bgnl),
        c: std::slice::from_raw_parts(c, bgnl),
        d_skip: opt(d_skip, d.dim),
        z: opt(z, bdl),
        delta_bias: opt(delta_bias, d.dim),
        delta_softplus: delta_softplus != 0,
        reverse: reverse != 0,
    };
    let out_slice = std::slice::from_raw_parts_mut(out, bdl);
    let mut last_slice = if last_state.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts_mut(last_state, bdn))
    };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        selective_scan_with_options(
            &scan_dims,
            &input,
            out_slice,
            last_slice.as_deref_mut(),
            ScanOptions { backend, threading },
        )
    }));

    match result {
        Ok(Ok(())) => ARM_SCAN_OK,
        Ok(Err(ScanError::BackendUnavailable(_))) => ARM_SCAN_ERR_BACKEND_UNAVAILABLE,
        Ok(Err(_)) => ARM_SCAN_ERR_INVALID_DIMS,
        Err(_) => ARM_SCAN_ERR_PANIC,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercise the full C path from Rust: hand-computed single step.
    #[test]
    fn ffi_roundtrip_single_step() {
        let dims = ArmScanDims {
            batch: 1,
            dim: 1,
            len: 1,
            state: 1,
            groups: 1,
        };
        let (u, dt, a, b, c) = ([0.5_f32], [0.1_f32], [-2.0_f32], [1.5_f32], [2.0_f32]);
        let mut out = [0.0_f32];
        let mut last = [0.0_f32];
        let code = unsafe {
            arm_scan_selective_scan_f32(
                &dims,
                u.as_ptr(),
                dt.as_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                0,
                0,
                0,
                out.as_mut_ptr(),
                last.as_mut_ptr(),
            )
        };
        assert_eq!(code, ARM_SCAN_OK);
        let h = 0.1 * 0.5 * 1.5; // dt*u*b (state starts at 0)
        assert!((out[0] - (2.0 * h) as f32).abs() < 1e-6);
        assert!((last[0] - h as f32).abs() < 1e-6);
    }

    /// `reverse` across the C ABI, hand-computed. Two timesteps, N=1, no
    /// softplus: a backward scan consumes t=1 first (state starts at zero at
    /// the END), so out[1] sees only its own input and out[0] carries the decay
    /// from t=1. Output still lands at index t — the layout never flips.
    #[test]
    fn ffi_reverse_two_steps() {
        let dims = ArmScanDims {
            batch: 1,
            dim: 1,
            len: 2,
            state: 1,
            groups: 1,
        };
        let (u, dt, a, b, c) = (
            [1.0_f32, 2.0],
            [0.1_f32, 0.2],
            [-2.0_f32],
            [1.0_f32, 3.0],
            [1.0_f32, 1.0],
        );
        let mut out = [0.0_f32; 2];
        let mut last = [0.0_f32; 1];
        let code = unsafe {
            arm_scan_selective_scan_f32(
                &dims,
                u.as_ptr(),
                dt.as_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0, // delta_softplus
                1, // reverse
                0,
                0,
                out.as_mut_ptr(),
                last.as_mut_ptr(),
            )
        };
        assert_eq!(code, ARM_SCAN_OK);

        // backward: h after t=1, then after t=0
        let h1 = 0.2_f32 * 2.0 * 3.0; // dt*u*b at t=1 (state starts at 0)
        let h0 = (0.1_f32 * -2.0).exp() * h1 + 0.1 * 1.0 * 1.0;
        assert!((out[1] - h1).abs() < 1e-6, "out[1]={} want {h1}", out[1]);
        assert!((out[0] - h0).abs() < 1e-6, "out[0]={} want {h0}", out[0]);
        // last_state under reverse is the state after consuming t == 0
        assert!((last[0] - h0).abs() < 1e-6);
    }

    #[test]
    fn ffi_rejects_null_and_bad_enum() {
        let dims = ArmScanDims {
            batch: 1,
            dim: 1,
            len: 1,
            state: 1,
            groups: 1,
        };
        let x = [0.0_f32];
        let mut out = [0.0_f32];
        let code = unsafe {
            arm_scan_selective_scan_f32(
                &dims,
                std::ptr::null(),
                x.as_ptr(),
                x.as_ptr(),
                x.as_ptr(),
                x.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                0,
                0,
                0,
                out.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(code, ARM_SCAN_ERR_NULL_POINTER);

        let code = unsafe {
            arm_scan_selective_scan_f32(
                &dims,
                x.as_ptr(),
                x.as_ptr(),
                x.as_ptr(),
                x.as_ptr(),
                x.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                0,
                7, // bad backend enum
                0,
                out.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(code, ARM_SCAN_ERR_BAD_ENUM);
    }
}
