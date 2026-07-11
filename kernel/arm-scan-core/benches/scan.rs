//! Criterion microbenchmarks for the selective scan.
//!
//! Shapes mirror the plan's benchmark ladder: a mamba-130m-like layer and a
//! smaller shape for quick iteration. Every optimization phase (NEON,
//! chunked, rayon) reruns these to show its contribution.

use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use arm_scan_core::{selective_scan_with_backend, Backend, ScanDims, ScanInput};

/// Deterministic pseudo-random fill (xorshift), no rand dependency.
fn fill(v: &mut [f32], mut seed: u32, lo: f32, hi: f32) {
    for x in v.iter_mut() {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        *x = lo + (hi - lo) * (seed as f32 / u32::MAX as f32);
    }
}

fn bench_scan(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("selective_scan");
    group
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(10);

    // On aarch64 this shows the scalar -> NEON rung of the optimization
    // ladder; on x86 both rows are the scalar fallback.
    let backends = [(Backend::Scalar, "scalar"), (Backend::Auto, "auto")];

    // (label, B, D, L, N)
    let shapes = [
        ("small_d64_l128", 2, 64, 128, 16),
        ("mamba130m_layer_l512", 1, 1536, 512, 16),
    ];

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

        // element-throughput: B*D*L cells, each doing N exp+FMA lanes
        group.throughput(Throughput::Elements(bdl as u64));
        for (backend, bname) in backends {
            group.bench_with_input(BenchmarkId::new(bname, label), &dims, |bch, dims| {
                bch.iter(|| {
                    selective_scan_with_backend(dims, &input, &mut out, None, backend).unwrap();
                    std::hint::black_box(&out);
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_scan);
criterion_main!(benches);
