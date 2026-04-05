# Kepler OpenCL Fork Guide

## Purpose

This fork is intentionally narrowed to one GPU path:

- modern `llama.cpp` frontend and GGUF support,
- legacy NVIDIA / Kepler execution through OpenCL 1.2,
- CLBlast-backed `MUL_MAT`,
- a growing set of OpenCL 1.2 FP32 kernels (`legacy_core.cl`) for norms, RoPE, softmax, masks, and elementwise ops,
- CPU fallback for everything not explicitly validated or outside the supported op matrix (see `docs/backend/OPENCL.md`).

The goal is not to preserve upstream's broad backend matrix. The goal is to keep this fork buildable and maintainable for the Quadro `K3100M` class of hardware.

## What Was Changed

Build-facing changes in this fork:

- added a legacy OpenCL profile with `GGML_OPENCL_LEGACY_NVIDIA=ON`,
- added optional CLBlast integration with `GGML_OPENCL_USE_CLBLAST=ON`,
- added fork-only CMake presets in `CMakePresets.json`,
- added a dedicated GitHub Actions workflow in `.github/workflows/fork-kepler-opencl-build.yml`,
- removed upstream build/release workflows that target incompatible backends or packaging paths.

Runtime behavior changes:

- NVIDIA OpenCL devices are no longer rejected outright,
- OpenCL 1.2 is accepted for the legacy NVIDIA path,
- after the CLBlast + `legacy_core` bootstrap, the backend still attempts the full `load_cl_kernels` compile pass at OpenCL C 1.2; builds that fail are skipped (no abort) so subgroup-heavy or driver-fragile sources simply leave null kernels,
- validated `MUL_MAT` workloads use CLBlast when the driver accepts CLBlast’s generated programs; otherwise the same path falls back to a native FP32 GEMM kernel on GPU (optional `GGML_OPENCL_LEGACY_NATIVE_GEMM=1` forces that),
- `GET_ROWS` (embeddings) for F32 / Q4_0 / Q6_K runs on GPU in `legacy_core.cl` so small models like Gemma / Qwen are not stuck on CPU for the gather alone,
- when a single FP32 dequant buffer would exceed `CL_DEVICE_MAX_MEM_ALLOC_SIZE`, `MUL_MAT` is split into column slices (still on GPU) instead of failing with OpenCL allocation errors,
- ops beyond CLBlast `MUL_MAT` use the same scheduling matrix as the standard OpenCL backend when the corresponding kernel compiled; unsupported or uncompiled ops stay on CPU (log gaps with `GGML_OPENCL_LEGACY_OP_INVENTORY=1`).
- OpenCL devices in legacy mode report `[fork_kepler_opencl]` in `ggml_backend_dev_description`. Without `cl_khr_fp16`, `[opencl_legacy_no_fp16]` is added; `llama_context` promotes K/V to F32 and disables Flash Attention so attention uses the decomposed GPU path where supported.
- KV cache is no longer forced to CPU solely for F16 on tagged fork devices (F32 K/V is used when the device has no FP16 extension).
- Shared `cl_context` lifetime: the backend does **not** call `clReleaseContext` when the last `ggml_backend` is freed (default). The first probe creates one shared context for all enumerated devices; `ggml_cl2_init` returns a cached `backend_ctx`, so releasing the context during model load/teardown left stale handles and could trigger `CL_INVALID_CONTEXT` (-34) on legacy NVIDIA drivers. The context is released at process exit. Set `GGML_OPENCL_RELEASE_CONTEXT=1` only if you want explicit release in the same process (reload without restart may fail).

## Supported Build Path

This fork now treats the following as the canonical build configurations:

- `fork-kepler-linux-release`
- `fork-kepler-windows-release`

Both presets force these backend decisions:

- `GGML_OPENCL=ON`
- `GGML_OPENCL_LEGACY_NVIDIA=ON`
- `GGML_OPENCL_USE_CLBLAST=ON`
- `GGML_OPENCL_USE_ADRENO_KERNELS=OFF`
- `GGML_OPENCL_TARGET_VERSION=120`
- `GGML_CUDA=OFF`
- `GGML_HIP=OFF`
- `GGML_MUSA=OFF`
- `GGML_VULKAN=OFF`
- `GGML_SYCL=OFF`
- `GGML_OPENVINO=OFF`
- `GGML_WEBGPU=OFF`

## GitHub Actions

The fork now builds through:

- `.github/workflows/fork-kepler-opencl-build.yml`

What it does automatically:

1. Builds Linux artifacts on `ubuntu-22.04`.
2. Builds Windows artifacts on `windows-2022`.
3. Configures both jobs through the fork-only presets.
4. Fetches CLBlast automatically through CMake when it is not already installed.
5. Uploads the built binaries as workflow artifacts.

What it does not try to do:

- benchmark unrelated backends,
- build CUDA, Vulkan, SYCL, Metal, HIP, MUSA, WebGPU, or OpenVINO targets,
- produce upstream release packaging formats,
- validate GPU execution on GitHub-hosted hardware.

## How To Use Built Artifacts

After a successful workflow run:

1. Download the `fork-kepler-opencl-windows` artifact for your machine.
2. Extract the archive.
3. Keep the executable and any bundled `.dll` files in the same folder.
4. Make sure the NVIDIA driver on the target machine still exposes OpenCL for the Kepler GPU.
5. Run the binaries the same way you normally use `llama.cpp`.

Operational notes:

- GPU layer offload still follows normal `llama.cpp` usage, e.g. `-ngl`.
- If you need to force OpenCL device selection, use `GGML_OPENCL_PLATFORM` and `GGML_OPENCL_DEVICE`.
- With `GGML_OPENCL_LEGACY_NVIDIA` builds, when **`GGML_OPENCL_PLATFORM` is unset**, init **prefers the NVIDIA OpenCL platform** if one exists with a GPU (hybrid systems often enumerate Intel first; without this, the iGPU hit the OpenCL 2.0 requirement and the dGPU was never probed).
- If a graph section is not supported by the legacy OpenCL path, it should fall back to CPU rather than trying to execute an unsafe kernel.

## When Pulling New Upstream Changes

Use this checklist whenever you sync from upstream:

1. Merge or rebase upstream into your fork.
2. Re-run `.github/workflows/fork-kepler-opencl-build.yml`.
3. Inspect conflicts and behavior in these files first:
   - `ggml/src/ggml-opencl/ggml-opencl.cpp`
   - `ggml/src/ggml-opencl/CMakeLists.txt`
   - `ggml/CMakeLists.txt`
   - `CMakePresets.json`
4. Re-check whether upstream changed:
   - `ggml-backend` scheduler behavior,
   - `MUL_MAT` tensor/layout assumptions,
   - OpenCL buffer initialization,
   - build option names or backend registration.
5. Update `docs/backend/OPENCL.md` and `CHANGELOG.md` if the fork path changes again.

## Maintenance Policy For This Fork

For this fork, build-system simplicity wins over upstream feature breadth.

That means:

- source code for unrelated backends may still exist in-tree,
- but CI and documented build paths should stay focused on the Kepler OpenCL target,
- if an upstream build path is not useful for this fork, it should stay removed or disabled.
