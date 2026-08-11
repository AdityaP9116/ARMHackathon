# Benchmark results

Generated 2026-08-11 04:58 UTC by `bench/render_results.py` — do not edit numbers by hand.

Surface tags per BASELINE_TEST_PLAN.md: dedicated Arm hardware is headline-grade; shared CI runners are provisional; x86 hosts exercise the scalar backend only.

## host tag: `graviton-c8g`

### bidirectional `sweep-len` — `bidir_graviton-c8g.json`

- host: Linux-6.17.0-1017-aws-aarch64-with-glibc2.39 (aarch64, 64 cpus), torch 2.13.0+cu130, reps=10

| shape B,D,L,N | fused ms | two-call ms | eager ms | compile ms | ×eager | ×compile | exp-sharing |
|---|---|---|---|---|---|---|---|
| 1,768,128,16 | 0.67 | 0.95 | 19.80 | 35.07 (compile 2s) | 29.74× | 52.66× | **1.43×** |
| 1,768,512,16 | 0.81 | 1.18 | 79.69 | 67.51 (compile 5s) | 98.01× | 83.03× | **1.45×** |
| 1,768,1024,16 | 1.04 | 1.53 | 186.80 | — | 178.76× | — | **1.47×** |
| 1,768,2048,16 | 1.38 | 2.22 | 395.79 | — | 287.07× | — | **1.61×** |
| 1,768,4096,16 | 2.42 | 3.78 | 809.33 | — | 334.40× | — | **1.56×** |
| 1,768,8192,16 | 4.31 | 6.91 | 1759.41 | — | 407.76× | — | **1.60×** |

`exp-sharing` is fused-vs-two-call: Pass A (discretize + `exp`) computed once instead of per direction.

### diffusion app — `diffusion_graviton-c8g.json`

- host: Linux-6.17.0-1017-aws-aarch64-with-glibc2.39 (aarch64), torch 2.13.0+cu130 (64 threads), reps=3
- prior: `untrained (timing only)`

| grid | params | per-NFE s | peak RSS MB | NFE=18 | NFE=69 | NFE=256 |
|---|---|---|---|---|---|---|
| 384x320 | 0.8 M | **0.436** | 2985 | 7.9 s | 30.1 s | 112 s |
| 192x160 | 0.8 M | **0.115** | 2985 | 2.1 s | 7.9 s | 29 s |
| 128x128 | 0.8 M | **0.071** | 2985 | 1.3 s | 4.9 s | 18 s |
| 64x64 | 0.8 M | **0.059** | 2985 | 1.1 s | 4.1 s | 15 s |

_Quality rows need a trained prior (`--checkpoint`); timing above is prior-independent._

### e2e `state-spaces/mamba-130m-hf` — `e2e_p128_graviton-c8g.json`

- host: Linux-6.17.0-1017-aws-aarch64-with-glibc2.39 (aarch64), torch 2.13.0+cu130 (64 threads), git 88caa09, 2026-08-11T04:48:17Z
- prompt 128 tok + 32 new, greedy, tokens identical: **True**

| | prefill ms | decode tok/s | total s |
|---|---|---|---|
| unpatched | 275.9 | 37.04 | 1.139 |
| patched | 175.5 | 38.72 | 1.012 |
| **speedup** | **1.57×** | — | **1.13×** |

### e2e `state-spaces/mamba-130m-hf` — `e2e_p2048_graviton-c8g.json`

- host: Linux-6.17.0-1017-aws-aarch64-with-glibc2.39 (aarch64), torch 2.13.0+cu130 (64 threads), git 88caa09, 2026-08-11T04:49:18Z
- prompt 2048 tok + 32 new, greedy, tokens identical: **True**

| | prefill ms | decode tok/s | total s |
|---|---|---|---|
| unpatched | 3634.5 | 41.48 | 4.403 |
| patched | 358.3 | 40.18 | 1.169 |
| **speedup** | **10.14×** | — | **3.77×** |

### op `basic` — `op_basic_graviton-c8g.json`

- host: Linux-6.17.0-1017-aws-aarch64-with-glibc2.39 (aarch64, 64 cpus), torch 2.13.0+cu130 (64 threads)
- git 88caa09, 2026-08-11T04:31:34Z, reps=10

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,768,128,16 | 9.68 | 17.12 (compile 43s) | 0.17 | 57.64× | 101.94× | 4.05e-06 |
| 1,768,512,16 | 39.93 | 32.87 (compile 111s) | 0.30 | 132.29× | 108.91× | 5.25e-06 |
| 1,768,2048,16 | 196.36 | — | 1.01 | 194.10× | — | 4.29e-06 |
| 8,1536,1024,16 | 640.02 | — | 5.68 | 112.67× | — | 9.06e-06 |

### op `sweep-batch` — `op_sweep-batch_graviton-c8g.json`

- host: Linux-6.17.0-1017-aws-aarch64-with-glibc2.39 (aarch64, 64 cpus), torch 2.13.0+cu130 (64 threads)
- git 88caa09, 2026-08-11T04:36:34Z, reps=10

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,1536,1024,16 | 161.11 | — | 0.92 | 174.69× | — | 4.65e-06 |
| 4,1536,1024,16 | 522.63 | — | 3.01 | 173.70× | — | 5.84e-06 |
| 8,1536,1024,16 | 639.31 | — | 5.66 | 112.88× | — | 9.06e-06 |

### op `sweep-dim` — `op_sweep-dim_graviton-c8g.json`

- host: Linux-6.17.0-1017-aws-aarch64-with-glibc2.39 (aarch64, 64 cpus), torch 2.13.0+cu130 (64 threads)
- git 88caa09, 2026-08-11T04:36:29Z, reps=10

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,256,512,16 | 25.95 | — | 0.17 | 153.29× | — | 9.39e-06 |
| 1,768,512,16 | 38.48 | — | 0.29 | 134.67× | — | 5.25e-06 |
| 1,1536,512,16 | 61.73 | — | 0.43 | 141.95× | — | 6.68e-06 |
| 1,3072,512,16 | 112.54 | — | 0.73 | 154.09× | — | 4.05e-06 |

### op `sweep-len` — `op_sweep-len_graviton-c8g.json`

- host: Linux-6.17.0-1017-aws-aarch64-with-glibc2.39 (aarch64, 64 cpus), torch 2.13.0+cu130 (64 threads)
- git 88caa09, 2026-08-11T04:34:25Z, reps=10

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,768,64,16 | 5.39 | 7.72 (compile 28s) | 0.15 | 35.18× | 50.37× | 2.15e-06 |
| 1,768,128,16 | 9.72 | 16.04 (compile 1s) | 0.16 | 59.49× | 98.12× | 4.05e-06 |
| 1,768,256,16 | 18.44 | 22.81 (compile 61s) | 0.20 | 90.47× | 111.94× | 4.89e-06 |
| 1,768,512,16 | 39.45 | 31.51 (compile 5s) | 0.29 | 137.75× | 110.02× | 5.25e-06 |
| 1,768,1024,16 | 95.86 | — | 0.62 | 154.77× | — | 8.11e-06 |
| 1,768,2048,16 | 189.83 | — | 1.02 | 186.10× | — | 4.29e-06 |
| 1,768,4096,16 | 412.61 | — | 1.73 | 238.34× | — | 5.25e-06 |
| 1,768,8192,16 | 867.45 | — | 3.18 | 272.84× | — | 7.92e-06 |

### ss2d (2D cross-scan) — `ss2d_graviton-c8g.json`

- host: Linux-6.17.0-1017-aws-aarch64-with-glibc2.39 (aarch64), torch 2.13.0+cu130 (64 threads), reps=5

Traversal-pair path vs the legacy four-forward-scans formulation. **Same kernel on both sides**, so the ratio is attributable to the restructuring, not the backend.

| case | pair ms | legacy ms | pair× | scan× | non-scan % | eager ms | compile ms |
|---|---|---|---|---|---|---|---|
| `L1_384x320_in96_b1` | 78.0 | 70.7 | **0.91×** | 0.87× | 24.6% | — | — |
| `L1_384x320_in96_b4` | 326.4 | 339.3 | **1.04×** | 1.03× | 22.6% | — | — |
| `L2_192x160_in192_b1` | 34.8 | 33.0 | **0.95×** | 1.03× | 46.1% | — | — |
| `L2_192x160_in192_b4` | 116.8 | 112.2 | **0.96×** | 0.92× | 33.0% | — | — |
| `mini_96x80_in96_b1` | 6.5 | 8.0 | **1.24×** | 1.77× | 72.0% | 435.9 | skipped |
| `tiny_32x32_in96_b1` | 4.3 | 9.1 | **2.09×** | 1.79× | 86.7% | 66.0 | 9439.0 |

- traversal-pair rewrite: **0.96× geomean** on the production shapes (block total)
- fully fused `selective_scan_2d` (P1-7): **JUSTIFIED** by the 15% non-scan-overhead rule

## host tag: `graviton-c8g-t1`

### op `scaling-point` — `op_scaling_graviton-c8g_t1.json`

- host: Linux-6.17.0-1017-aws-aarch64-with-glibc2.39 (aarch64, 64 cpus), torch 2.13.0+cu130 (64 threads)
- git 88caa09, 2026-08-11T04:36:55Z, reps=7

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,1536,512,16 | 60.67 | — | 17.55 | 3.46× | — | 6.68e-06 |

## host tag: `graviton-c8g-t16`

### op `scaling-point` — `op_scaling_graviton-c8g_t16.json`

- host: Linux-6.17.0-1017-aws-aarch64-with-glibc2.39 (aarch64, 64 cpus), torch 2.13.0+cu130 (64 threads)
- git 88caa09, 2026-08-11T04:37:06Z, reps=7

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,1536,512,16 | 60.92 | — | 1.22 | 50.13× | — | 6.68e-06 |

## host tag: `graviton-c8g-t2`

### op `scaling-point` — `op_scaling_graviton-c8g_t2.json`

- host: Linux-6.17.0-1017-aws-aarch64-with-glibc2.39 (aarch64, 64 cpus), torch 2.13.0+cu130 (64 threads)
- git 88caa09, 2026-08-11T04:36:58Z, reps=7

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,1536,512,16 | 62.02 | — | 8.76 | 7.08× | — | 6.68e-06 |

## host tag: `graviton-c8g-t32`

### op `scaling-point` — `op_scaling_graviton-c8g_t32.json`

- host: Linux-6.17.0-1017-aws-aarch64-with-glibc2.39 (aarch64, 64 cpus), torch 2.13.0+cu130 (64 threads)
- git 88caa09, 2026-08-11T04:37:08Z, reps=7

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,1536,512,16 | 60.87 | — | 0.74 | 81.80× | — | 6.68e-06 |

## host tag: `graviton-c8g-t4`

### op `scaling-point` — `op_scaling_graviton-c8g_t4.json`

- host: Linux-6.17.0-1017-aws-aarch64-with-glibc2.39 (aarch64, 64 cpus), torch 2.13.0+cu130 (64 threads)
- git 88caa09, 2026-08-11T04:37:00Z, reps=7

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,1536,512,16 | 64.00 | — | 4.40 | 14.56× | — | 6.68e-06 |

## host tag: `graviton-c8g-t64`

### op `scaling-point` — `op_scaling_graviton-c8g_t64.json`

- host: Linux-6.17.0-1017-aws-aarch64-with-glibc2.39 (aarch64, 64 cpus), torch 2.13.0+cu130 (64 threads)
- git 88caa09, 2026-08-11T04:37:11Z, reps=7

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,1536,512,16 | 60.17 | — | 0.45 | 132.66× | — | 6.68e-06 |

## host tag: `graviton-c8g-t8`

### op `scaling-point` — `op_scaling_graviton-c8g_t8.json`

- host: Linux-6.17.0-1017-aws-aarch64-with-glibc2.39 (aarch64, 64 cpus), torch 2.13.0+cu130 (64 threads)
- git 88caa09, 2026-08-11T04:37:03Z, reps=7

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,1536,512,16 | 61.80 | — | 2.23 | 27.70× | — | 6.68e-06 |

## host tag: `ip-172-31-38-148`

### unrecognised result — `m3-kernel-graviton-r15.json`

- `kind` = `mamba3`, top-level keys: `abi, cases, git_sha, kind, machine, platform, reps, tag, threads, torch, warmup`
- No renderer matched. The JSON is intact; add a renderer in `render_results.py` rather than re-running the benchmark.

### unrecognised result — `m3-kernel-graviton.json`

- `kind` = `mamba3`, top-level keys: `abi, cases, git_sha, kind, machine, platform, reps, tag, threads, torch, warmup`
- No renderer matched. The JSON is intact; add a renderer in `render_results.py` rather than re-running the benchmark.

## host tag: `untagged`

### FAILED TO RENDER — `m3-2d-graviton.json`

- `kind` = `op`, renderer raised `KeyError: 'env'`
- The JSON is intact; fix the renderer, re-run this script. Do not re-run the benchmark.

### FAILED TO RENDER — `m3-lm-mimo-graviton.json`

- `kind` = `op`, renderer raised `KeyError: 'env'`
- The JSON is intact; fix the renderer, re-run this script. Do not re-run the benchmark.

### FAILED TO RENDER — `m3-lm-siso-graviton.json`

- `kind` = `op`, renderer raised `KeyError: 'env'`
- The JSON is intact; fix the renderer, re-run this script. Do not re-run the benchmark.

### FAILED TO RENDER — `m3-noncausal-graviton-r15.json`

- `kind` = `op`, renderer raised `KeyError: 'env'`
- The JSON is intact; fix the renderer, re-run this script. Do not re-run the benchmark.

### FAILED TO RENDER — `m3-noncausal-graviton.json`

- `kind` = `op`, renderer raised `KeyError: 'env'`
- The JSON is intact; fix the renderer, re-run this script. Do not re-run the benchmark.

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

- `criterion_graviton-c8g.txt`
- `criterion_windows-i9.txt`
