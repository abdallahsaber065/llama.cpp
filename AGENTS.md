# Instructions for This Fork

This repository is a **private fork** focused on a narrow goal, not on upstream open source contribution workflow.

## Fork Goal

The main goal of this fork is:

- keep modern `llama.cpp` frontend and GGUF usability,
- support **legacy NVIDIA / Kepler OpenCL 1.2** hardware,
- use **CLBlast-backed `MUL_MAT`** where validated,
- allow **safe CPU fallback** for unsupported graph sections,
- keep the build, docs, and release flow simple and consistent for this fork.

This fork should optimize for the owner's machine and workflow first, especially the Quadro `K3100M` class path, not for upstream feature breadth.

## Primary Supported Path

Treat these as the canonical configurations for this fork:

- CMake presets:
  - `fork-kepler-linux-release`
  - `fork-kepler-windows-release`
- Workflow:
  - `.github/workflows/fork-kepler-opencl-build.yml` (sole active workflow; others are `*.yml.disabled`)
  - Automatic **GitHub Release** on each successful `master` push with tag `kepler-opencl-r<run>-a<attempt>-<sha7>`
- Backend mode (fork presets):
  - `GGML_OPENCL=OFF`
  - `GGML_OPENCL_KEPLER=ON`
  - `GGML_OPENCL_LEGACY_NVIDIA=ON`
  - `GGML_OPENCL_USE_CLBLAST=ON`
  - `GGML_OPENCL_TARGET_VERSION=120`

Do not optimize this fork around CUDA, Vulkan, SYCL, Metal, HIP, MUSA, WebGPU, or OpenVINO unless the user explicitly asks.

## OpenCL-Kepler / Quadro K3100M rules

When changing the Kepler path, keep **scheduler claims** aligned with **what actually compiled** on the device:

1. **Canonical binary**: Prefer validating behavior against **`fork-kepler-linux-release`** / **`fork-kepler-windows-release`** (backend name **`OpenCL-Kepler`**, device **`GPUOpenCLKepler`**). Stock **`GGML_OPENCL`** builds are secondary for this fork’s stated goal.
2. **Relaxed `load_cl_kernels`**: On failed builds, `build_program_from_source` returns **`nullptr`**. Never call **`clCreateKernel`** on a null program. Pair every optional kernel with a **`supports_op`** (or dispatch) check so graph scheduling does not route to a missing kernel (avoids `GGML_ASSERT` / undefined behavior).
3. **Typical K3100M caps**: Often **no `cl_khr_fp16`**, **no `cl_khr_subgroups`**, and **~1 GiB `CL_DEVICE_MAX_MEM_ALLOC_SIZE`**. Long **`kernel compile error`** logs during init can be **normal** on this stack; the process should continue and leave **null** kernels for failed programs.
4. **`GGML_OP_FLASH_ATTN_EXT`**: Only treat as supported when the matching per-`(dk,dv)` / `n_q` flash kernel handle is **non-null** after init. FP16 flash sources are **not** built when **`fp16_support`** is false.
5. **“GPU-only” vs fork default**: The fork still allows **CPU fallback** when `supports_op` is false (conservative). Maximizing GPU fraction on Kepler means **more OpenCL 1.2–safe kernels** and accurate **`supports_op`**, not forcing dispatch without a compiled kernel.

## Most Important Files

When debugging or extending the fork, inspect these first:

- `ggml/src/ggml-opencl/ggml-opencl.cpp`
- `ggml/src/ggml-opencl-kepler/ggml-opencl-kepler.cpp`
- `ggml/src/ggml-opencl/CMakeLists.txt`
- `ggml/src/ggml-opencl-kepler/CMakeLists.txt`
- `ggml/src/ggml-opencl/kernels/*.cl`
- `ggml/CMakeLists.txt`
- `CMakePresets.json`
- `.github/workflows/fork-kepler-opencl-build.yml`
- `src/llama-kv-cache.cpp`
- `src/llama-memory-recurrent.cpp`
- `common/arg.cpp`
- `docs/kepler-opencl-fork.md`
- `docs/backend/OPENCL.md`
- `CHANGELOG.md`

## Fork-Specific Priorities

Prefer these principles:

1. Keep the legacy NVIDIA / Kepler path working end-to-end.
2. Prefer conservative fallback over risky GPU execution.
3. Prefer small, targeted fixes over broad upstream-style refactors.
4. Keep runtime behavior understandable and debuggable.
5. Keep CI, release logic, and docs aligned with actual fork behavior.

## Required Update Workflow

After **any** meaningful fix, feature, behavior change, CI change, runtime change, or release-flow change, update the corresponding files automatically as part of the same task.

Minimum required follow-up:

- Update `CHANGELOG.md`.
- Update `docs/kepler-opencl-fork.md` if the fork workflow, scope, releases, or supported path changed.
- Update `docs/backend/OPENCL.md` if runtime behavior, supported devices, supported dtypes, build flags, or usage instructions changed.
- Update `.github/workflows/fork-kepler-opencl-build.yml` if build/release packaging expectations changed.
- Update `docs/fork-merge-pins.md` or `.gitattributes` if fork merge-pin paths or CI disable policy changed.
- Update `AGENTS.md` if the persistent fork workflow or maintenance rules changed.

Do not leave code behavior and docs out of sync.

## CHANGELOG Rules

`CHANGELOG.md` is the single top-level change record for this fork.

- Add a concise entry for every meaningful repository change.
- Do not create extra summary files just to explain a task.
- If release automation depends on version headings, preserve that format.
- When a task changes runtime behavior, mention the practical effect, not just the file touched.

## Documentation Rules

Keep documentation minimal and consolidated:

- put user-facing docs in `docs/`,
- extend existing docs before creating new docs,
- avoid redundant summary files,
- document only behavior that users or future maintainers actually need.

For this fork, the default docs to update are:

- `docs/kepler-opencl-fork.md`
- `docs/backend/OPENCL.md`

## Runtime Debugging Guidance

When investigating runtime failures on the target Kepler path, check these areas in order:

1. OpenCL device selection:
   - `GGML_OPENCL_PLATFORM`
   - `GGML_OPENCL_DEVICE`
2. OpenCL init logs:
   - platform chosen
   - device chosen
   - driver version
   - FP16 support
   - workgroup size
3. Legacy compatibility gating:
   - `GGML_OPENCL_LEGACY_NVIDIA`
   - CLBlast path
   - disabled subgroup-heavy kernels
4. Scheduler/backend placement:
   - tensor preallocation
   - KV cache placement
   - `SET_ROWS`
   - CPU fallback vs unsafe GPU placement
5. Kernel compile/runtime issues:
   - OpenCL 1.2 restrictions
   - FP16 kernel support
   - PTX/driver compiler failures

## Build and Release Guidance

This fork uses one primary workflow:

- `.github/workflows/fork-kepler-opencl-build.yml`

Expected behavior:

- build Linux and Windows artifacts,
- keep artifact packaging usable as extracted binaries,
- create a GitHub Release only when the configured versioned release condition is met,
- use the fork's simplified release flow, not upstream release packaging logic.

If build or release behavior changes, update both:

- workflow file
- relevant docs and changelog

## AI Agent Behavior For This Fork

AI agents working in this repo should:

- treat this as a fork-maintenance project, not an upstream contribution task,
- keep answers concise and practical,
- favor implementation and maintenance help over upstream policy advice,
- preserve the fork's narrow Kepler/OpenCL focus,
- automatically update the related docs/changelog/workflow files when a fix changes behavior,
- avoid introducing broad new backend scope unless explicitly requested.

## Do Not Default To

Avoid these by default:

- upstream contributor-policy guidance,
- broad multi-backend refactors,
- removing the CPU fallback safety behavior,
- assuming modern OpenCL 2.x/3.x capabilities on legacy NVIDIA,
- adding documentation sprawl,
- changing release/version flow casually without updating docs and changelog.

## Quick Reference

Load these first when relevant:

- `docs/kepler-opencl-fork.md`
- `docs/backend/OPENCL.md`
- `CHANGELOG.md`
- `CMakePresets.json`
- `.github/workflows/fork-kepler-opencl-build.yml`
- `ggml/src/ggml-opencl/ggml-opencl.cpp`
- `src/llama-kv-cache.cpp`
