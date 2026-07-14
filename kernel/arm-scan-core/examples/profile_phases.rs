//! Phase-breakdown profiler for the NEON selective scan.
//!
//! Runs the instrumented single-threaded kernel (`scan_profiled`) over a set
//! of mamba-shaped problems and prints, per shape, how kernel time splits
//! across transpose / discretize / exp / projection / recurrence / epilogue.
//! This is the free "where is the bottleneck" tool — see PROFILING.md.
//!
//!   cargo run --release --example profile_phases --features profiling
//!
//! Meaningful only on aarch64 (real NEON). On other hosts it explains what to
//! run instead.

#[cfg(all(target_arch = "aarch64", feature = "profiling"))]
fn main() {
    use arm_scan_core::{scan_profiled, ScanDims, ScanInput};

    // Deterministic xorshift fill (no rand dependency); mirrors benches/scan.rs.
    fn fill(v: &mut [f32], mut seed: u32, lo: f32, hi: f32) {
        for x in v.iter_mut() {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *x = lo + (hi - lo) * (seed as f32 / u32::MAX as f32);
        }
    }

    // (label, batch, dim, len, state). N=16 is the fast path the profiler
    // supports. L sweeps so the transpose/DRAM share can be seen growing.
    let shapes = [
        ("mamba130m_L128", 1, 1536, 128, 16),
        ("mamba130m_L512", 1, 1536, 512, 16),
        ("mamba130m_L2048", 1, 1536, 2048, 16),
        ("batch8_L1024", 8, 1536, 1024, 16),
    ];

    println!("# NEON selective-scan phase profile (single-threaded)");
    println!("# host: aarch64  | read RELATIVE %, not absolute ns");
    println!("# transpose is a SERIAL prologue: at C cores its wall share ~×C\n");

    for (label, batch, dim, len, state) in shapes {
        let dims = ScanDims {
            batch,
            dim,
            len,
            state,
            groups: 1,
        };
        let bdl = batch * dim * len;
        let bgnl = batch * state * len;

        let mut u = vec![0.0_f32; bdl];
        let mut delta = vec![0.0_f32; bdl];
        let mut a = vec![0.0_f32; dim * state];
        let mut b = vec![0.0_f32; bgnl];
        let mut c = vec![0.0_f32; bgnl];
        let mut z = vec![0.0_f32; bdl];
        let mut d_skip = vec![0.0_f32; dim];
        let mut bias = vec![0.0_f32; dim];
        fill(&mut u, 1, -3.0, 3.0);
        fill(&mut delta, 2, -2.0, 2.0);
        fill(&mut a, 3, -16.0, -0.5);
        fill(&mut b, 4, -3.0, 3.0);
        fill(&mut c, 5, -3.0, 3.0);
        fill(&mut z, 6, -3.0, 3.0);
        fill(&mut d_skip, 7, -1.0, 1.0);
        fill(&mut bias, 8, -6.0, -3.0);

        let input = ScanInput {
            u: &u,
            delta: &delta,
            a: &a,
            b: &b,
            c: &c,
            d_skip: Some(&d_skip),
            z: Some(&z),
            delta_bias: Some(&bias),
            delta_softplus: true,
        };
        let mut out = vec![0.0_f32; bdl];

        // Warm up (page-in, caches), then take the best of several runs so a
        // scheduler hiccup doesn't skew the split.
        let mut best = scan_profiled(&dims, &input, &mut out);
        for _ in 0..4 {
            let t = scan_profiled(&dims, &input, &mut out);
            if t.total_ns() < best.total_ns() {
                best = t;
            }
        }
        std::hint::black_box(&out);

        let total = best.total_ns().max(1) as f64;
        let pct = |x: u128| 100.0 * x as f64 / total;
        println!("## {label}  (B={batch} D={dim} L={len} N={state})");
        println!("   phase          {:>12}   {:>6}", "ns", "%");
        let rows = [
            ("transpose", best.transpose_ns),
            ("discretize", best.discretize_ns),
            ("exp", best.exp_ns),
            ("projection", best.proj_ns),
            ("recurrence", best.recurrence_ns),
            ("epilogue", best.epilogue_ns),
        ];
        for (name, ns) in rows {
            println!("   {name:<13} {ns:>12}   {:>5.1}", pct(ns));
        }
        println!(
            "   {:<13} {:>12}   {:>5.1}\n",
            "TOTAL",
            best.total_ns(),
            100.0
        );
    }

    println!("Interpretation (see IMPROVEMENT_IDEAS.md):");
    println!("  exp dominant        -> §3.1 cheaper exp / §3.2 SVE FEXPA");
    println!("  recurrence dominant -> §3.3 chain-breaking (esp. Graviton4)");
    println!("  transpose large     -> §2.1 layout flag / §4.1 workspace reuse");
    println!("  grows with L        -> §4.2 cache-blocking (protects long-context)");
}

#[cfg(not(all(target_arch = "aarch64", feature = "profiling")))]
fn main() {
    eprintln!(
        "profile_phases requires aarch64 + the `profiling` feature.\n\
         On this host (non-Arm or feature off) the NEON path does not run.\n\n\
         Run it on real Arm silicon:\n  \
         cargo run --release --example profile_phases --features profiling\n\n\
         For free Arm hardware, trigger the `profile` GitHub Actions workflow\n\
         (Actions tab -> Profile kernel -> Run workflow) and download the\n\
         `kernel-profile` artifact. See PROFILING.md."
    );
}
