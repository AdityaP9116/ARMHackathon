## baseline suite, linux-arm64 CI runner (provisional)

# Benchmark results

Generated 2026-07-13 15:36 UTC by `bench/render_results.py` — do not edit numbers by hand.

Surface tags per BASELINE_TEST_PLAN.md: dedicated Arm hardware is headline-grade; shared CI runners are provisional; x86 hosts exercise the scalar backend only.

## host tag: `ci-arm64`

### e2e `state-spaces/mamba-130m-hf` — `e2e_p128_ci-arm64.json`

- host: Linux-6.17.0-1018-azure-aarch64-with-glibc2.39 (aarch64), torch 2.13.0+cu130 (4 threads), git ce07edd, 2026-07-13T15:33:49Z
- prompt 128 tok + 32 new, greedy, tokens identical: **True**

| | prefill ms | decode tok/s | total s |
|---|---|---|---|
| unpatched | 508.3 | 36.16 | 1.393 |
| patched | 271.9 | 36.08 | 1.159 |
| **speedup** | **1.87×** | — | **1.20×** |

### e2e `state-spaces/mamba-130m-hf` — `e2e_p2048_ci-arm64.json`

- host: Linux-6.17.0-1018-azure-aarch64-with-glibc2.39 (aarch64), torch 2.13.0+cu130 (4 threads), git ce07edd, 2026-07-13T15:36:55Z
- prompt 2048 tok + 32 new, greedy, tokens identical: **True**

| | prefill ms | decode tok/s | total s |
|---|---|---|---|
| unpatched | 7559.3 | 331.89 | 7.696 |
| patched | 3588.2 | 163.52 | 3.784 |
| **speedup** | **2.11×** | — | **2.03×** |

### e2e `state-spaces/mamba-130m-hf` — `e2e_p512_ci-arm64.json`

- host: Linux-6.17.0-1018-azure-aarch64-with-glibc2.39 (aarch64), torch 2.13.0+cu130 (4 threads), git ce07edd, 2026-07-13T15:34:36Z
- prompt 512 tok + 32 new, greedy, tokens identical: **True**

| | prefill ms | decode tok/s | total s |
|---|---|---|---|
| unpatched | 1911.4 | 43.55 | 2.634 |
| patched | 958.5 | 44.38 | 1.671 |
| **speedup** | **1.99×** | — | **1.58×** |

### op `basic` — `op_basic_ci-arm64.json`

- host: Linux-6.17.0-1018-azure-aarch64-with-glibc2.39 (aarch64, 4 cpus), torch 2.13.0+cu130 (4 threads)
- git ce07edd, 2026-07-13T15:25:39Z, reps=10

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,768,128,16 | 13.83 | 3.18 (compile 65s) | 0.96 | 14.42× | 3.32× | 1.91e-06 |
| 1,768,512,16 | 71.66 | 13.62 (compile 159s) | 3.27 | 21.88× | 4.16× | 2.74e-06 |
| 1,768,2048,16 | 267.80 | — | 13.07 | 20.48× | — | 2.15e-06 |
| 8,1536,1024,16 | 839.30 | — | 99.33 | 8.45× | — | 3.81e-06 |

### op `sweep-batch` — `op_sweep-batch_ci-arm64.json`

- host: Linux-6.17.0-1018-azure-aarch64-with-glibc2.39 (aarch64, 4 cpus), torch 2.13.0+cu130 (4 threads)
- git ce07edd, 2026-07-13T15:32:39Z, reps=10

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,1536,1024,16 | 212.00 | — | 12.82 | 16.53× | — | 2.38e-06 |
| 4,1536,1024,16 | 495.53 | — | 49.73 | 9.96× | — | 3.81e-06 |
| 8,1536,1024,16 | 814.33 | — | 99.87 | 8.15× | — | 3.81e-06 |

### op `sweep-dim` — `op_sweep-dim_ci-arm64.json`

- host: Linux-6.17.0-1018-azure-aarch64-with-glibc2.39 (aarch64, 4 cpus), torch 2.13.0+cu130 (4 threads)
- git ce07edd, 2026-07-13T15:32:31Z, reps=10

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,256,512,16 | 37.10 | — | 1.22 | 30.36× | — | 1.49e-06 |
| 1,768,512,16 | 68.89 | — | 3.34 | 20.65× | — | 2.74e-06 |
| 1,1536,512,16 | 117.19 | — | 6.45 | 18.17× | — | 2.15e-06 |
| 1,3072,512,16 | 143.07 | — | 12.80 | 11.18× | — | 1.91e-06 |

### op `sweep-len` — `op_sweep-len_ci-arm64.json`

- host: Linux-6.17.0-1018-azure-aarch64-with-glibc2.39 (aarch64, 4 cpus), torch 2.13.0+cu130 (4 threads)
- git ce07edd, 2026-07-13T15:29:48Z, reps=10

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,768,64,16 | 6.87 | 1.43 (compile 37s) | 0.55 | 12.38× | 2.57× | 1.43e-06 |
| 1,768,128,16 | 14.39 | 2.94 (compile 2s) | 0.94 | 15.32× | 3.13× | 1.91e-06 |
| 1,768,256,16 | 28.77 | 6.25 (compile 86s) | 1.73 | 16.66× | 3.62× | 1.91e-06 |
| 1,768,512,16 | 69.79 | — | 3.29 | 21.19× | — | 2.74e-06 |
| 1,768,1024,16 | 132.02 | — | 6.51 | 20.29× | — | 3.81e-06 |
| 1,768,2048,16 | 272.23 | — | 12.97 | 20.99× | — | 2.15e-06 |
| 1,768,4096,16 | 542.11 | — | 26.16 | 20.73× | — | 2.86e-06 |
| 1,768,8192,16 | 1121.34 | — | 52.23 | 21.47× | — | 2.62e-06 |

## host tag: `ci-arm64-t1`

### op `scaling-point` — `op_scaling_ci-arm64_t1.json`

- host: Linux-6.17.0-1018-azure-aarch64-with-glibc2.39 (aarch64, 4 cpus), torch 2.13.0+cu130 (4 threads)
- git ce07edd, 2026-07-13T15:33:06Z, reps=7

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,1536,512,16 | 111.30 | — | 25.08 | 4.44× | — | 2.15e-06 |

## host tag: `ci-arm64-t2`

### op `scaling-point` — `op_scaling_ci-arm64_t2.json`

- host: Linux-6.17.0-1018-azure-aarch64-with-glibc2.39 (aarch64, 4 cpus), torch 2.13.0+cu130 (4 threads)
- git ce07edd, 2026-07-13T15:33:10Z, reps=7

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,1536,512,16 | 112.18 | — | 12.63 | 8.88× | — | 2.15e-06 |

## host tag: `ci-arm64-t4`

### op `scaling-point` — `op_scaling_ci-arm64_t4.json`

- host: Linux-6.17.0-1018-azure-aarch64-with-glibc2.39 (aarch64, 4 cpus), torch 2.13.0+cu130 (4 threads)
- git ce07edd, 2026-07-13T15:33:14Z, reps=7

| shape B,D,L,N | eager ms | compile ms | kernel ms | ×eager | ×compile | max_abs_err |
|---|---|---|---|---|---|---|
| 1,1536,512,16 | 117.79 | — | 6.47 | 18.21× | — | 2.15e-06 |

## raw criterion ladders

- `criterion_ci-arm64.txt`
