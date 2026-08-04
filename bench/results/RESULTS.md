# Benchmark results

Generated 2026-08-04 17:02 UTC by `bench/render_results.py` — do not edit numbers by hand.

Surface tags per BASELINE_TEST_PLAN.md: dedicated Arm hardware is headline-grade; shared CI runners are provisional; x86 hosts exercise the scalar backend only.

## host tag: `Aditya-Patra`

### diffusion app — `diffusion_windows-dev.json`

- host: Windows-11-10.0.26200-SP0 (AMD64), torch 2.13.0+cpu (16 threads), reps=1
- prior: `untrained (timing only)`

| grid | params | per-NFE s | peak RSS MB | NFE=18 | NFE=69 | NFE=256 |
|---|---|---|---|---|---|---|
| 48x32 | 0.8 M | **0.087** | — | 1.6 s | 6.0 s | 22 s |
| 384x320 | 0.8 M | **26.786** | — | 482.1 s | 1848.2 s | 6857 s |

**Cost per reconstruction** at $2.9000/h (`c8g.16xlarge`):

| grid | NFE=18 | NFE=69 | NFE=256 |
|---|---|---|---|
| 48x32 | $0.0013 | $0.0048 | $0.0179 |
| 384x320 | $0.3884 | $1.4889 | $5.5239 |

_Quality rows need a trained prior (`--checkpoint`); timing above is prior-independent._

## host tag: `windows-dev-clean`

### ss2d (2D cross-scan) — `ss2d_windows-dev.json`

- host: Windows-11-10.0.26200-SP0 (AMD64), torch 2.13.0+cpu (16 threads), reps=3

Traversal-pair path vs the legacy four-forward-scans formulation. **Same kernel on both sides**, so the ratio is attributable to the restructuring, not the backend.

| case | pair ms | legacy ms | pair× | scan× | non-scan % | eager ms | compile ms |
|---|---|---|---|---|---|---|---|
| `L1_384x320_in96_b1` | 984.0 | 1739.5 | **1.77×** | 1.81× | 7.2% | — | — |
| `L1_384x320_in96_b4` | 3971.0 | 7218.1 | **1.82×** | 1.90× | 13.8% | — | — |
| `L2_192x160_in192_b1` | 468.8 | 850.7 | **1.81×** | 1.87× | 7.2% | — | — |
| `L2_192x160_in192_b4` | 1863.5 | 3386.8 | **1.82×** | 1.88× | 9.4% | — | — |
| `mini_96x80_in96_b1` | 63.4 | 118.6 | **1.87×** | 2.02× | 12.6% | 637.3 | — |
| `tiny_32x32_in96_b1` | 10.2 | 17.8 | **1.74×** | 1.81× | 11.6% | 87.3 | — |

- traversal-pair rewrite: **1.80× geomean** on the production shapes (block total)
- fully fused `selective_scan_2d` (P1-7): **not justified** by the 15% non-scan-overhead rule

## host tag: `windows-i9`

### e2e `state-spaces/mamba-130m-hf` — `e2e_p128_windows-i9.json`

- host: Windows-10-10.0.26200-SP0 (AMD64), torch 2.11.0.dev20260208+cu128 (24 threads), git cea6d1e, 2026-07-13T15:08:31Z
- prompt 128 tok + 32 new, greedy, tokens identical: **True**

| | prefill ms | decode tok/s | total s |
|---|---|---|---|
| unpatched | 970.7 | 21.48 | 2.460 |
| patched | 326.2 | 20.34 | 1.903 |
| **speedup** | **2.98×** | — | **1.29×** |

### e2e `state-spaces/mamba-130m-hf` — `e2e_p2048_windows-i9.json`

- host: Windows-10-10.0.26200-SP0 (AMD64), torch 2.11.0.dev20260208+cu128 (24 threads), git cea6d1e, 2026-07-13T15:13:05Z
- prompt 2048 tok + 32 new, greedy, tokens identical: **True**

| | prefill ms | decode tok/s | total s |
|---|---|---|---|
| unpatched | 8029.9 | 29.07 | 9.131 |
| patched | 2808.3 | 36.53 | 3.584 |
| **speedup** | **2.86×** | — | **2.55×** |

### e2e `state-spaces/mamba-130m-hf` — `e2e_p512_windows-i9.json`

- host: Windows-10-10.0.26200-SP0 (AMD64), torch 2.11.0.dev20260208+cu128 (24 threads), git cea6d1e, 2026-07-13T15:10:03Z
- prompt 512 tok + 32 new, greedy, tokens identical: **True**

| | prefill ms | decode tok/s | total s |
|---|---|---|---|
| unpatched | 3601.0 | 21.33 | 5.094 |
| patched | 447.2 | 26.22 | 1.648 |
| **speedup** | **8.05×** | — | **3.09×** |

### op `basic` — `op_basic_windows-i9.json`

- host: Windows-10-10.0.26200-SP0 (AMD64, 32 cpus), torch 2.11.0.dev20260208+cu128 (24 threads)
- git 4c18403, 2026-07-13T14:59:23Z, reps=10

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,768,128,16 | 18.78 | unavailable | 1.69 | 11.09× | — | 9.54e-07 |
| 1,768,512,16 | 127.23 | unavailable | 4.18 | 30.42× | — | 1.91e-06 |
| 1,768,2048,16 | 633.37 | — | 9.92 | 63.82× | — | 1.91e-06 |
| 8,1536,1024,16 | 1307.22 | — | 73.57 | 17.77× | — | 2.38e-06 |

### op `sweep-batch` — `op_sweep-batch_windows-i9.json`

- host: Windows-10-10.0.26200-SP0 (AMD64, 32 cpus), torch 2.11.0.dev20260208+cu128 (24 threads)
- git cea6d1e, 2026-07-13T15:04:32Z, reps=10

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,1536,1024,16 | 550.57 | — | 10.24 | 53.76× | — | 1.91e-06 |
| 4,1536,1024,16 | 961.57 | — | 38.02 | 25.29× | — | 2.86e-06 |
| 8,1536,1024,16 | 1269.96 | — | 74.33 | 17.09× | — | 2.38e-06 |

### op `sweep-dim` — `op_sweep-dim_windows-i9.json`

- host: Windows-10-10.0.26200-SP0 (AMD64, 32 cpus), torch 2.11.0.dev20260208+cu128 (24 threads)
- git cea6d1e, 2026-07-13T15:04:16Z, reps=10

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,256,512,16 | 76.46 | — | 1.05 | 72.96× | — | 1.91e-06 |
| 1,768,512,16 | 125.28 | — | 2.64 | 47.51× | — | 1.91e-06 |
| 1,1536,512,16 | 219.51 | — | 5.02 | 43.77× | — | 1.91e-06 |
| 1,3072,512,16 | 284.93 | — | 9.71 | 29.35× | — | 1.91e-06 |

### op `sweep-len` — `op_sweep-len_windows-i9.json`

- host: Windows-10-10.0.26200-SP0 (AMD64, 32 cpus), torch 2.11.0.dev20260208+cu128 (24 threads)
- git 4c18403, 2026-07-13T15:01:23Z, reps=10

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,768,64,16 | 12.93 | unavailable | 1.42 | 9.12× | — | 1.43e-06 |
| 1,768,128,16 | 25.75 | unavailable | 1.76 | 14.64× | — | 9.54e-07 |
| 1,768,256,16 | 51.05 | unavailable | 2.16 | 23.62× | — | 1.91e-06 |
| 1,768,512,16 | 120.74 | unavailable | 3.28 | 36.78× | — | 1.91e-06 |
| 1,768,1024,16 | 314.75 | — | 5.31 | 59.31× | — | 1.91e-06 |
| 1,768,2048,16 | 647.47 | — | 9.96 | 65.02× | — | 1.91e-06 |
| 1,768,4096,16 | 1315.36 | — | 20.71 | 63.51× | — | 2.86e-06 |
| 1,768,8192,16 | 2721.49 | — | 37.65 | 72.29× | — | 2.86e-06 |

### ss2d (2D cross-scan) — `ss2d_windows-i9.json`

- host: Windows-10-10.0.26200-SP0 (AMD64), torch 2.11.0.dev20260208+cu128 (? threads), reps=3

Traversal-pair path vs the legacy four-forward-scans formulation. **Same kernel on both sides**, so the ratio is attributable to the restructuring, not the backend.

| case | pair ms | legacy ms | pair× | scan× | non-scan % | eager ms | compile ms |
|---|---|---|---|---|---|---|---|
| `L1_384x320_in96_b1` | 646.3 | — | — | — | 22.4% | — | — |
| `L1_384x320_in96_b4` | 2636.2 | — | — | — | 21.4% | — | — |
| `L2_192x160_in192_b1` | 305.2 | — | — | — | 23.7% | — | — |
| `L2_192x160_in192_b4` | 1173.7 | — | — | — | 24.7% | — | — |
| `mini_96x80_in96_b1` | 43.8 | — | — | — | 33.5% | 1466.1 | — |
- fully fused `selective_scan_2d` (P1-7): **JUSTIFIED** by the 15% non-scan-overhead rule

## host tag: `windows-i9-t1`

### op `scaling-point` — `op_scaling_windows-i9_t1.json`

- host: Windows-10-10.0.26200-SP0 (AMD64, 32 cpus), torch 2.11.0.dev20260208+cu128 (24 threads)
- git cea6d1e, 2026-07-13T15:05:59Z, reps=7

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,1536,512,16 | 215.79 | — | 84.19 | 2.56× | — | 1.91e-06 |

## host tag: `windows-i9-t16`

### op `scaling-point` — `op_scaling_windows-i9_t16.json`

- host: Windows-10-10.0.26200-SP0 (AMD64, 32 cpus), torch 2.11.0.dev20260208+cu128 (24 threads)
- git cea6d1e, 2026-07-13T15:06:32Z, reps=7

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,1536,512,16 | 270.76 | — | 13.31 | 20.34× | — | 1.91e-06 |

## host tag: `windows-i9-t2`

### op `scaling-point` — `op_scaling_windows-i9_t2.json`

- host: Windows-10-10.0.26200-SP0 (AMD64, 32 cpus), torch 2.11.0.dev20260208+cu128 (24 threads)
- git cea6d1e, 2026-07-13T15:06:08Z, reps=7

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,1536,512,16 | 216.72 | — | 41.59 | 5.21× | — | 1.91e-06 |

## host tag: `windows-i9-t32`

### op `scaling-point` — `op_scaling_windows-i9_t32.json`

- host: Windows-10-10.0.26200-SP0 (AMD64, 32 cpus), torch 2.11.0.dev20260208+cu128 (24 threads)
- git cea6d1e, 2026-07-13T15:06:42Z, reps=7

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,1536,512,16 | 272.20 | — | 9.71 | 28.04× | — | 1.91e-06 |

## host tag: `windows-i9-t4`

### op `scaling-point` — `op_scaling_windows-i9_t4.json`

- host: Windows-10-10.0.26200-SP0 (AMD64, 32 cpus), torch 2.11.0.dev20260208+cu128 (24 threads)
- git cea6d1e, 2026-07-13T15:06:16Z, reps=7

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,1536,512,16 | 221.77 | — | 27.18 | 8.16× | — | 1.91e-06 |

## host tag: `windows-i9-t8`

### op `scaling-point` — `op_scaling_windows-i9_t8.json`

- host: Windows-10-10.0.26200-SP0 (AMD64, 32 cpus), torch 2.11.0.dev20260208+cu128 (24 threads)
- git cea6d1e, 2026-07-13T15:06:24Z, reps=7

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,1536,512,16 | 227.54 | — | 15.49 | 14.69× | — | 1.91e-06 |

## raw criterion ladders

- `criterion_windows-i9.txt`
