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
//! Nullable: d_skip, z, delta_bias, last_state, h0. Everything else non-null.
//!   h0 : (batch, dim, state) initial SSM state; null = zero-initialized.

use std::os::raw::c_int;

use arm_scan_core::{
    mamba3_scan_with_options, selective_scan_bidirectional, selective_scan_with_state, Backend,
    Mamba3Dims, Mamba3Input, ScanDims, ScanError, ScanInput, ScanOptions, Threading,
};

/// ABI version. The Python loader checks this before calling anything else.
/// Bump on any signature or semantic change to the entry points.
///
/// 4: `h0` (resumable initial state) and `reverse` (backward-in-time traversal)
///    were developed on separate branches, each bumping to 3. Both are in 4.
/// 5: added `arm_scan_selective_scan_bidirectional_f32` (fused two-direction).
/// 6: added `arm_scan_mamba3_scan_f32` (Mamba-3 SISO, its own dims struct).
#[no_mangle]
pub extern "C" fn arm_scan_abi_version() -> u32 {
    6
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
    h0: *const f32,
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
    let h0_slice = if h0.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts(h0, bdn))
    };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        selective_scan_with_state(
            &scan_dims,
            &input,
            out_slice,
            last_slice.as_deref_mut(),
            h0_slice,
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

/// Fused bidirectional scan: one set of inputs, both the forward output
/// (`out_fwd`) and the backward output (`out_bwd`), computing the shared,
/// direction-independent Pass A (discretize + exp) once. Semantically equal to
/// two [`arm_scan_selective_scan_f32`] calls (forward, then `reverse=1`) but
/// sharing the ~85% exp cost; see BIDIRECTIONAL_SPEEDUP_IDEAS.md.
///
/// No `reverse`/`h0` params: direction is inherent, and both directions seed
/// from zero. `last_fwd`/`last_bwd` are nullable but must be null together or
/// non-null together.
///
/// # Safety
/// Same contract as [`arm_scan_selective_scan_f32`]: every non-null pointer
/// references a buffer of the element count implied by `dims`, valid for the
/// call, non-overlapping. `out_fwd`/`out_bwd` are `(batch, dim, len)`;
/// `last_fwd`/`last_bwd` are `(batch, dim, state)`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn arm_scan_selective_scan_bidirectional_f32(
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
    backend: c_int,
    threading: c_int,
    out_fwd: *mut f32,
    out_bwd: *mut f32,
    last_fwd: *mut f32,
    last_bwd: *mut f32,
) -> c_int {
    if dims.is_null()
        || u.is_null()
        || delta.is_null()
        || a.is_null()
        || b.is_null()
        || c.is_null()
        || out_fwd.is_null()
        || out_bwd.is_null()
    {
        return ARM_SCAN_ERR_NULL_POINTER;
    }
    let (Some(backend), Some(threading)) = (backend_from(backend), threading_from(threading))
    else {
        return ARM_SCAN_ERR_BAD_ENUM;
    };

    let d = &*dims;
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
        reverse: false, // ignored by the fused path (both directions produced)
    };
    let out_fwd_slice = std::slice::from_raw_parts_mut(out_fwd, bdl);
    let out_bwd_slice = std::slice::from_raw_parts_mut(out_bwd, bdl);
    let last_fwd_slice = if last_fwd.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts_mut(last_fwd, bdn))
    };
    let last_bwd_slice = if last_bwd.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts_mut(last_bwd, bdn))
    };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        selective_scan_bidirectional(
            &scan_dims,
            &input,
            out_fwd_slice,
            out_bwd_slice,
            last_fwd_slice,
            last_bwd_slice,
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

/// Dimensions for a Mamba-3 SISO scan.
///
/// A separate struct from [`ArmScanDims`] rather than an overload: Mamba-3's
/// state is a `(dv, dqk)` matrix per head, not a vector per channel, and its
/// tensor set is disjoint from Mamba-1's. One struct serving both would be half
/// ignored on every call.
#[repr(C)]
pub struct ArmMamba3Dims {
    pub batch: usize,
    pub heads: usize,
    /// Head dim of `v`/`out` — the state matrix's rows.
    pub dv: usize,
    /// Head dim of `q`/`k` — the state matrix's columns. Must be even.
    pub dqk: usize,
    pub len: usize,
}

/// Run the Mamba-3 SISO selective scan.
///
/// `backend`: 0 = auto, 1 = scalar, 2 = neon. `threading`: 0 = auto,
/// 1 = sequential, 2 = rayon. `reverse`: nonzero walks the sequence backward.
///
/// **`out` is head-major** `(batch, heads, len, dv)` while the inputs are
/// time-major — see `arm_scan_core::mamba3` for why. The Python wrapper
/// permutes.
///
/// **`trap` is pre-sigmoid**: the kernel applies it, matching upstream, so a
/// caller cannot accidentally apply it twice.
///
/// **`cos`/`sin` are precomputed** by the caller's angle pre-pass
/// (`theta = cumsum(tanh(angle) * PI * dt)`), mirroring upstream's split between
/// `angle_dt_fwd` and `mamba3_siso_fwd`.
///
/// Layout contract:
/// ```text
///   q, k            (batch, len, 1, dqk)
///   v, z            (batch, len, heads, dv)
///   adt, dt, trap   (batch, heads, len)
///   cos, sin        (batch, len, heads, dqk/2)
///   q_bias, k_bias  (heads, dqk)
///   d_skip          (heads,)
///   out             (batch, heads, len, dv)
///   last_state      (batch, heads, dv, dqk)
///   last_bx         (batch, heads, dqk)
/// ```
/// Nullable: `d_skip`, `z`, `last_state`, `last_bx`. The last two must be null
/// together or non-null together — a resumed scan needs both, and silently
/// dropping one would produce a plausible but wrong continuation.
///
/// # Safety
/// Same contract as [`arm_scan_selective_scan_f32`]: every non-null pointer
/// references a buffer of exactly the element count implied by `dims` and the
/// layout above, valid for the call, non-overlapping.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn arm_scan_mamba3_scan_f32(
    dims: *const ArmMamba3Dims,
    q: *const f32,
    k: *const f32,
    v: *const f32,
    adt: *const f32,
    dt: *const f32,
    trap: *const f32,
    q_bias: *const f32,
    k_bias: *const f32,
    cos: *const f32,
    sin: *const f32,
    d_skip: *const f32,
    z: *const f32,
    reverse: c_int,
    backend: c_int,
    threading: c_int,
    out: *mut f32,
    last_state: *mut f32,
    last_bx: *mut f32,
) -> c_int {
    if dims.is_null()
        || q.is_null()
        || k.is_null()
        || v.is_null()
        || adt.is_null()
        || dt.is_null()
        || trap.is_null()
        || q_bias.is_null()
        || k_bias.is_null()
        || cos.is_null()
        || sin.is_null()
        || out.is_null()
    {
        return ARM_SCAN_ERR_NULL_POINTER;
    }
    // A half-supplied carry pair is a caller bug, not a defaulting opportunity.
    if last_state.is_null() != last_bx.is_null() {
        return ARM_SCAN_ERR_NULL_POINTER;
    }
    let (Some(backend), Some(threading)) = (backend_from(backend), threading_from(threading))
    else {
        return ARM_SCAN_ERR_BAD_ENUM;
    };

    let d = &*dims;
    if !d.dqk.is_multiple_of(2) {
        return ARM_SCAN_ERR_INVALID_DIMS;
    }
    // Overflow-checked element counts before any slice is formed.
    let mul = |a: usize, b: usize| a.checked_mul(b);
    let (Some(blqk), Some(bhl), Some(hqk)) = (
        mul(d.batch, d.len).and_then(|x| mul(x, d.dqk)),
        mul(d.batch, d.heads).and_then(|x| mul(x, d.len)),
        mul(d.heads, d.dqk),
    ) else {
        return ARM_SCAN_ERR_INVALID_DIMS;
    };
    let (Some(blhv), Some(blhr), Some(bhvq), Some(bhq)) = (
        mul(d.batch, d.len)
            .and_then(|x| mul(x, d.heads))
            .and_then(|x| mul(x, d.dv)),
        mul(d.batch, d.len)
            .and_then(|x| mul(x, d.heads))
            .and_then(|x| mul(x, d.dqk / 2)),
        mul(d.batch, d.heads)
            .and_then(|x| mul(x, d.dv))
            .and_then(|x| mul(x, d.dqk)),
        mul(d.batch, d.heads).and_then(|x| mul(x, d.dqk)),
    ) else {
        return ARM_SCAN_ERR_INVALID_DIMS;
    };

    let m3_dims = Mamba3Dims {
        batch: d.batch,
        heads: d.heads,
        dv: d.dv,
        dqk: d.dqk,
        len: d.len,
    };
    let opt = |p: *const f32, n: usize| {
        if p.is_null() {
            None
        } else {
            Some(std::slice::from_raw_parts(p, n))
        }
    };
    let input = Mamba3Input {
        q: std::slice::from_raw_parts(q, blqk),
        k: std::slice::from_raw_parts(k, blqk),
        v: std::slice::from_raw_parts(v, blhv),
        adt: std::slice::from_raw_parts(adt, bhl),
        dt: std::slice::from_raw_parts(dt, bhl),
        trap: std::slice::from_raw_parts(trap, bhl),
        q_bias: std::slice::from_raw_parts(q_bias, hqk),
        k_bias: std::slice::from_raw_parts(k_bias, hqk),
        cos: std::slice::from_raw_parts(cos, blhr),
        sin: std::slice::from_raw_parts(sin, blhr),
        d_skip: opt(d_skip, d.heads),
        z: opt(z, blhv),
        reverse: reverse != 0,
    };
    let out_slice = std::slice::from_raw_parts_mut(out, blhv);
    let mut ls = if last_state.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts_mut(last_state, bhvq))
    };
    let mut lb = if last_bx.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts_mut(last_bx, bhq))
    };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        mamba3_scan_with_options(
            &m3_dims,
            &input,
            out_slice,
            ls.as_deref_mut(),
            lb.as_deref_mut(),
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
                std::ptr::null(),
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
                std::ptr::null(), // h0: zero-initialized
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

    /// Fused bidirectional across the C ABI, hand-computed. Two steps, N=1: the
    /// forward output is the ordinary scan; the backward output must match the
    /// `reverse` case above. Both from one call sharing Pass A.
    #[test]
    fn ffi_bidirectional_two_steps() {
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
        let mut out_fwd = [0.0_f32; 2];
        let mut out_bwd = [0.0_f32; 2];
        let code = unsafe {
            arm_scan_selective_scan_bidirectional_f32(
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
                0, // backend = auto
                0, // threading = auto
                out_fwd.as_mut_ptr(),
                out_bwd.as_mut_ptr(),
                std::ptr::null_mut(), // last_fwd
                std::ptr::null_mut(), // last_bwd
            )
        };
        assert_eq!(code, ARM_SCAN_OK);

        // forward: h at t=0, then t=1
        let f0 = 0.1_f32 * 1.0 * 1.0; // dt*u*b at t=0
        let f1 = (0.2_f32 * -2.0).exp() * f0 + 0.2 * 2.0 * 3.0;
        assert!((out_fwd[0] - f0).abs() < 1e-6, "out_fwd[0]={}", out_fwd[0]);
        assert!((out_fwd[1] - f1).abs() < 1e-6, "out_fwd[1]={}", out_fwd[1]);

        // backward: h at t=1, then t=0 (matches ffi_reverse_two_steps)
        let b1 = 0.2_f32 * 2.0 * 3.0;
        let b0 = (0.1_f32 * -2.0).exp() * b1 + 0.1 * 1.0 * 1.0;
        assert!((out_bwd[1] - b1).abs() < 1e-6, "out_bwd[1]={}", out_bwd[1]);
        assert!((out_bwd[0] - b0).abs() < 1e-6, "out_bwd[0]={}", out_bwd[0]);
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
                std::ptr::null(),
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
                std::ptr::null(),
            )
        };
        assert_eq!(code, ARM_SCAN_ERR_BAD_ENUM);
    }

    /// h0 flows through the C ABI: a 2-step scan split into two calls, with the
    /// first call's last_state fed back as h0, matches the one-shot scan.
    #[test]
    fn ffi_streaming_with_h0() {
        use std::ptr::{null, null_mut};
        let (u, dt, a, b, c) = (
            [0.5_f32, -0.3],
            [0.1_f32, 0.2],
            [-2.0_f32],
            [1.5_f32, 0.7],
            [2.0_f32, 1.1],
        );

        let dims_full = ArmScanDims {
            batch: 1,
            dim: 1,
            len: 2,
            state: 1,
            groups: 1,
        };
        let mut out_full = [0.0_f32; 2];
        let code = unsafe {
            arm_scan_selective_scan_f32(
                &dims_full,
                u.as_ptr(),
                dt.as_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                null(),
                null(),
                null(),
                0, // delta_softplus
                0, // reverse
                0,
                0,
                out_full.as_mut_ptr(),
                null_mut(),
                null(),
            )
        };
        assert_eq!(code, ARM_SCAN_OK);

        let dims1 = ArmScanDims {
            batch: 1,
            dim: 1,
            len: 1,
            state: 1,
            groups: 1,
        };
        // step 0: capture the intermediate state
        let mut out1 = [0.0_f32; 1];
        let mut state = [0.0_f32; 1];
        let code = unsafe {
            arm_scan_selective_scan_f32(
                &dims1,
                u.as_ptr(),
                dt.as_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                null(),
                null(),
                null(),
                0, // delta_softplus
                0, // reverse
                0,
                0,
                out1.as_mut_ptr(),
                state.as_mut_ptr(),
                null(),
            )
        };
        assert_eq!(code, ARM_SCAN_OK);

        // step 1: resume from `state` as h0
        let mut out2 = [0.0_f32; 1];
        let code = unsafe {
            arm_scan_selective_scan_f32(
                &dims1,
                u.as_ptr().add(1),
                dt.as_ptr().add(1),
                a.as_ptr(),
                b.as_ptr().add(1),
                c.as_ptr().add(1),
                null(),
                null(),
                null(),
                0, // delta_softplus
                0, // reverse
                0,
                0,
                out2.as_mut_ptr(),
                null_mut(),
                state.as_ptr(),
            )
        };
        assert_eq!(code, ARM_SCAN_OK);

        assert!(
            (out_full[0] - out1[0]).abs() < 1e-6,
            "{} vs {}",
            out_full[0],
            out1[0]
        );
        assert!(
            (out_full[1] - out2[0]).abs() < 1e-6,
            "{} vs {}",
            out_full[1],
            out2[0]
        );
    }

    /// The Mamba-3 entry point through the real C ABI, including the carry-pair
    /// rule. A two-step L=2 case is the smallest shape where the trapezoid's
    /// second term is actually exercised (at L=1 it has no successor).
    #[test]
    fn ffi_mamba3_two_steps() {
        let (b, h, dv, dqk, l) = (1usize, 2usize, 2usize, 4usize, 2usize);
        let dims = ArmMamba3Dims {
            batch: b,
            heads: h,
            dv,
            dqk,
            len: l,
        };
        let q = vec![0.3f32; b * l * dqk];
        let k = vec![0.2f32; b * l * dqk];
        let v = vec![1.0f32; b * l * h * dv];
        let adt = vec![-0.5f32; b * h * l];
        let dt = vec![0.1f32; b * h * l];
        let trap = vec![0.0f32; b * h * l];
        let qb = vec![0.0f32; h * dqk];
        let kb = vec![0.0f32; h * dqk];
        let cos = vec![1.0f32; b * l * h * dqk / 2];
        let sin = vec![0.0f32; b * l * h * dqk / 2];
        let mut out = vec![0.0f32; b * h * l * dv];
        let mut last_state = vec![0.0f32; b * h * dv * dqk];
        let mut last_bx = vec![0.0f32; b * h * dqk];

        let rc = unsafe {
            arm_scan_mamba3_scan_f32(
                &dims,
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                adt.as_ptr(),
                dt.as_ptr(),
                trap.as_ptr(),
                qb.as_ptr(),
                kb.as_ptr(),
                cos.as_ptr(),
                sin.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                0,
                1,
                out.as_mut_ptr(),
                last_state.as_mut_ptr(),
                last_bx.as_mut_ptr(),
            )
        };
        assert_eq!(rc, ARM_SCAN_OK);
        assert!(out.iter().all(|x| x.is_finite()));
        assert!(out.iter().any(|&x| x != 0.0), "output is all zeros");
        assert!(last_state.iter().any(|&x| x != 0.0), "state never updated");
        assert!(last_bx.iter().any(|&x| x != 0.0), "2-tap carry never set");
    }

    /// `last_state` and `last_bx` must be supplied together. Half a carry would
    /// resume a scan from a state that looks plausible and is wrong.
    #[test]
    fn ffi_mamba3_rejects_half_a_carry() {
        let dims = ArmMamba3Dims {
            batch: 1,
            heads: 1,
            dv: 2,
            dqk: 4,
            len: 1,
        };
        let one = vec![0.1f32; 64];
        let mut out = vec![0.0f32; 2];
        let mut state = vec![0.0f32; 8];
        let rc = unsafe {
            arm_scan_mamba3_scan_f32(
                &dims,
                one.as_ptr(),
                one.as_ptr(),
                one.as_ptr(),
                one.as_ptr(),
                one.as_ptr(),
                one.as_ptr(),
                one.as_ptr(),
                one.as_ptr(),
                one.as_ptr(),
                one.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                0,
                1,
                out.as_mut_ptr(),
                state.as_mut_ptr(),
                std::ptr::null_mut(), // last_bx omitted
            )
        };
        assert_eq!(rc, ARM_SCAN_ERR_NULL_POINTER);
    }

    /// An odd `dqk` has no meaning — RoPE rotates lane pairs.
    #[test]
    fn ffi_mamba3_rejects_odd_dqk() {
        let dims = ArmMamba3Dims {
            batch: 1,
            heads: 1,
            dv: 2,
            dqk: 3,
            len: 1,
        };
        let one = vec![0.1f32; 64];
        let mut out = vec![0.0f32; 2];
        let rc = unsafe {
            arm_scan_mamba3_scan_f32(
                &dims,
                one.as_ptr(),
                one.as_ptr(),
                one.as_ptr(),
                one.as_ptr(),
                one.as_ptr(),
                one.as_ptr(),
                one.as_ptr(),
                one.as_ptr(),
                one.as_ptr(),
                one.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                0,
                1,
                out.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, ARM_SCAN_ERR_INVALID_DIMS);
    }
}
