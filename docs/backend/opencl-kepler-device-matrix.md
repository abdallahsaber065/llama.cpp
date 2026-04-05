# OpenCL Kepler device matrix (Quadro K3100M class)

This document records **host OpenCL limits** for the fork’s primary hardware target. Fill in the **Measured** column from your machine (`clinfo` on Linux; Intel/NVIDIA OpenCL samples or GPU Caps Viewer on Windows).

## How to capture

- **Linux:** `clinfo -l` then `clinfo -d <platform>:<device>` for the NVIDIA GPU, or save full `clinfo > clinfo.txt`.
- **Windows:** use `clinfo` if installed, or vendor tools that report `CL_DEVICE_*` queries.

## Reference vs measured (mobile Kepler / Quadro K3100M class)

| Query / limit | Typical K3100M class (reference) | Measured (your GPU) |
|----------------|----------------------------------|---------------------|
| OpenCL platform | NVIDIA CUDA / ICD | |
| `CL_DEVICE_VERSION` | OpenCL 1.2 CUDA | |
| `CL_DEVICE_OPENCL_C_VERSION` | OpenCL C 1.2 | |
| `cl_khr_fp16` | Usually **absent** | |
| `cl_khr_subgroups` | Usually **absent** | |
| `CL_DEVICE_GLOBAL_MEM_SIZE` | ~4 GiB (VRAM) | |
| `CL_DEVICE_MAX_MEM_ALLOC_SIZE` | Often **~1 GiB** (critical for chunked dequant) | |
| `CL_DEVICE_MAX_WORK_GROUP_SIZE` | 1024 (verify) | |
| `CL_DEVICE_TYPE` | `CL_DEVICE_TYPE_GPU` | |

## Build linkage

The dedicated backend `ggml-opencl-kepler` is compiled with OpenCL **1.2** host API, **CLBlast** for GEMM-class work, `GGML_OPENCL_LEGACY_NVIDIA`, and `GGML_OPENCL_KEPLER_BUILD` (NVIDIA GPU + platform filter at probe). It reuses kernel sources under `ggml/src/ggml-opencl/kernels/`.

## Refresh policy

After merging upstream or changing OpenCL/CLBlast usage, re-run `clinfo` on the target GPU and update the **Measured** column so allocator and tiling assumptions stay honest.
