# llama.cpp for OpenCL

- [Background](#background)
- [Fork Guide](#fork-guide)
- [OS](#os)
- [Hardware](#hardware)
- [DataType Supports](#datatype-supports)
- [Model Preparation](#model-preparation)
- [CMake Options](#cmake-options)
- [Android](#android)
- [Windows 11 Arm64](#windows-11-arm64)
- [Linux](#Linux)
- [Known Issue](#known-issues)
- [TODO](#todo)

## Background

OpenCL (Open Computing Language) is an open, royalty-free standard for cross-platform, parallel programming of diverse accelerators found in supercomputers, cloud servers, personal computers, mobile devices and embedded platforms. OpenCL specifies a programming language (based on C99) for programming these devices and application programming interfaces (APIs) to control the platform and execute programs on the compute devices. Similar to CUDA, OpenCL has been widely used to program GPUs and is supported by most GPU vendors.

### Llama.cpp + OpenCL

The llama.cpp OpenCL backend is designed to enable llama.cpp on **Qualcomm Adreno GPU** firstly via OpenCL. Thanks to the portabilty of OpenCL, the OpenCL backend can also run on certain Intel GPUs such as those that do not have [SYCL](/docs/backend/SYCL.md) support although the performance is not optimal.

## Fork Guide

For the Kepler-focused fork workflow, GitHub Actions pipeline, and upstream-sync checklist, see [Kepler OpenCL Fork Guide](/docs/kepler-opencl-fork.md).

**Dedicated Kepler backend:** This fork can build `**ggml-opencl-kepler`** (`GGML_OPENCL_KEPLER=ON`) instead of stock `**ggml-opencl**` (`GGML_OPENCL=OFF`). The two options are **mutually exclusive** at configure time. Device matrix template: [opencl-kepler-device-matrix.md](opencl-kepler-device-matrix.md). Model-blind op parity checklist: [opencl-kepler-op-parity-matrix.md](opencl-kepler-op-parity-matrix.md).

## OS


| OS      | Status  | Verified                                 |
| ------- | ------- | ---------------------------------------- |
| Android | Support | Snapdragon 8 Gen 3, Snapdragon 8 Elite   |
| Windows | Support | Windows 11 Arm64 with Snapdragon X Elite |
| Linux   | Support | Ubuntu 22.04 WSL2 with Intel 12700H      |


## Hardware

### Adreno GPU

**Verified devices**


| Adreno GPU                      | Status  |
| ------------------------------- | ------- |
| Adreno 750 (Snapdragon 8 Gen 3) | Support |
| Adreno 830 (Snapdragon 8 Elite) | Support |
| Adreno X85 (Snapdragon X Elite) | Support |


> A6x GPUs with a recent driver and compiler are supported; they are usually found in IoT platforms.
> However, A6x GPUs in phones are likely not supported due to the outdated driver and compiler.

### Legacy NVIDIA / Kepler

This fork also includes a conservative compatibility path for older NVIDIA OpenCL stacks. It is intended for legacy GPUs such as Kepler-class mobile Quadro parts that cannot use the modern CUDA backend.

**Default platform (legacy builds):** If `GGML_OPENCL_PLATFORM` is unset and both Intel and NVIDIA OpenCL ICDs are present, the fork **prefers NVIDIA** so the first enumerated iGPU is not chosen by mistake (Intel OpenCL 1.2 fails the non-legacy OpenCL 2.0 gate and previously resulted in no usable GPU).

This path still loads CLBlast, `set_rows`, and a small **OpenCL 1.2 / FP32 / subgroup-free** program (`legacy_core.cl`) for common transformer ops. It then runs the same `**load_cl_kernels` pass** as the modern backend, compiled for OpenCL C 1.2: programs that fail to build are **skipped** (logged, null program/kernel) instead of aborting the process, so whatever your driver accepts can run on GPU. Scheduling uses the **standard** `supports_op` rules for those ops; `MUL_MAT` remains on the CLBlast + legacy GEMM path when `ggml_opencl_can_mul_mat_legacy` allows it. Anything the driver cannot compile or the backend does not advertise stays on CPU via the scheduler.

**`GGML_OP_FLASH_ATTN_EXT`:** After init, this op is only considered supported on GPU if the corresponding **flash-attention kernel** for the tensor head sizes `(dk, dv)` and sequence layout (`n_q == 1` uses the `*_q1` kernels) actually **compiled** (non-null `cl_kernel`). Failed program builds must not leave `supports_op` true for that variant, or dispatch would assert.

**CLBlast scope:** CLBlast provides BLAS-style routines (for example `GEMM`). Custom kernels cover the non-BLAS ops below; other `GGML_OP` types remain unsupported on GPU until implemented.

**Legacy core ops (FP32, contiguous where noted):**


| Op              | Notes                                                                                                                                                                                          |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MUL_MAT`       | CLBlast + dequant when it succeeds; otherwise native FP32 GEMM in `legacy_core.cl` (same GPU, slower). Set `GGML_OPENCL_LEGACY_NATIVE_GEMM=1` to skip CLBlast. Init runs `clblast::FillCache`. |
| `GET_ROWS`      | F32 / Q4_0 / Q6_K source rows → F32 dst; indices `I32`; contiguous; `ne[0]` divisible by 32 (Q4_0) or 256 (Q6_K).                                                                              |
| `SET_ROWS`      | Existing path; F16 only with `cl_khr_fp16`                                                                                                                                                     |
| `RMS_NORM`      | F32, contiguous                                                                                                                                                                                |
| `ROPE`          | Normal / NeoX only; not MROPE, vision, or IMROPE                                                                                                                                               |
| `SOFT_MAX`      | F32 activations; mask tensor must be F32 (no F16 mask)                                                                                                                                         |
| `DIAG_MASK_INF` | F32                                                                                                                                                                                            |
| `ADD`, `MUL`    | Same-shape F32                                                                                                                                                                                 |
| `SCALE`         | F32                                                                                                                                                                                            |
| `CPY`           | F32→F32, same shape, contiguous                                                                                                                                                                |
| `DUP`, `CONT`   | F32, same shape as source, contiguous                                                                                                                                                          |
| `UNARY`         | `SILU`, `GELU` (tanh approximation)                                                                                                                                                            |


**Device tags:** Legacy NVIDIA init on stock OpenCL appends `[fork_kepler_opencl]` to `ggml_backend_dev_description`. The dedicated **OpenCL-Kepler** backend appends `[opencl_kepler]` (same F16/KV behavior). Without `cl_khr_fp16`, `[opencl_legacy_no_fp16]` is also appended; `llama_context` then uses F32 K/V and turns off Flash Attention so graphs use the decomposed attention + `SOFT_MAX` path.

**Inventory:** Set `GGML_OPENCL_LEGACY_OP_INVENTORY=1` to log unsupported ops when the legacy backend declines a node (useful for model-by-model gap analysis).

**Per-buffer limit:** Many drivers cap a single `cl_mem` allocation (`CL_DEVICE_MAX_MEM_ALLOC_SIZE`, often 1 GiB on older NVIDIA OpenCL). The legacy path dequantizes weights to FP32 for CLBlast; when `ne × sizeof(float)` would exceed a conservative budget derived from that cap, it dequantizes and multiplies **column slices** of `src0` (along `ne01`) instead of one giant buffer. If a tensor cannot be tiled under sub-buffer alignment and dequant kernel work-group constraints, `MUL_MAT` is left for the CPU backend.

**Program release on Kepler:** During the optional full `load_cl_kernels` pass, failed compiles leave a null `cl_program`; drivers must not abort when releasing those handles. Legacy init uses a safe `clReleaseProgram` path (null skip, no assert) because some NVIDIA 411-era stacks return `CL_INVALID_PROGRAM` for `clReleaseProgram(NULL)`.

**Shared context lifetime (legacy NVIDIA):** One `cl_context` is created for all devices discovered in a single probe. Cached `ggml_backend_opencl_context` objects are reused without re-probing. Calling `clReleaseContext` when the last backend is freed therefore breaks a later open in the same process (stale `cl_context` / `CL_INVALID_CONTEXT` / assert in `CL_CHECK`). By default this fork **does not** release the shared context until process exit. Use `GGML_OPENCL_RELEASE_CONTEXT=1` only for debugging; expect possible failure if you load another model without restarting the process.

Current focus of the compatibility path:


| Area                 | Status                                                                    |
| -------------------- | ------------------------------------------------------------------------- |
| OpenCL 1.2 init path | Support                                                                   |
| CLBlast GEMM backend | Support                                                                   |
| `MUL_MAT` offload    | Support                                                                   |
| Legacy core F32 ops  | Partial (see table above); plus optional standard kernels if they compile |
| `MUL_MAT_ID` offload | GPU when `supports_op` / kernels match upstream OpenCL matrix             |
| Subgroup kernels     | Disabled                                                                  |


The OpenCL backend reports `ggml_backend_dev_description` as `device name (OpenCL C version)`, matching the init log line, plus optional fork tags (see above). Older logic that keyed off `OpenCL 1.2` in the description for KV placement is bypassed when `[fork_kepler_opencl]` or `[opencl_kepler]` is present (or when the device name is `GPUOpenCLKepler`).

## DataType Supports


| DataType | Status                     |
| -------- | -------------------------- |
| Q4_0     | Support                    |
| Q6_K     | Support, but not optimized |
| Q8_0     | Support                    |
| MXFP4    | Support                    |


For the legacy NVIDIA / Kepler compatibility path, the current GPU-offloaded `MUL_MAT` path is limited to:


| DataType | Status  |
| -------- | ------- |
| F32      | Support |
| F16      | Support |
| Q4_0     | Support |
| Q4_1     | Support |
| Q8_0     | Support |
| Q4_K     | Support |
| Q6_K     | Support |


## Model Preparation

You can refer to the general [llama-quantize tool](/tools/quantize/README.md) for steps to convert a model in Hugging Face safetensor format to GGUF with quantization.

Currently we support `Q4_0` quantization and have optimized for it. To achieve best performance on Adreno GPU, add `--pure` to `llama-quantize` (i.e., make all weights in `Q4_0`). For example,

```sh
./llama-quantize --pure ggml-model-qwen2.5-3b-f16.gguf ggml-model-qwen-3b-Q4_0.gguf Q4_0
```

Since `Q6_K` is also supported, `Q4_0` quantization without `--pure` will also work. However, the performance will be worse compared to pure `Q4_0` quantization.

### `MXFP4` MoE Models

OpenAI gpt-oss models are MoE models in `MXFP4`. The quantized model will be in `MXFP4_MOE`, a mixture of `MXFP4` and `Q8_0`.
For this quantization, there is no need to specify `--pure`.
For gpt-oss-20b model, you can directly [download](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) the quantized GGUF file in `MXFP4_MOE` from Hugging Face.

Although it is possible to quantize gpt-oss-20b model in pure `Q4_0` (all weights in `Q4_0`), it is not recommended since `MXFP4` has been optimized for MoE while `Q4_0` is not. In addition, accuracy should degrade with such pure `Q4_0` quantization.
Hence, using the default `MXFP4_MOE` quantization (see the link above) is recommended for this model.

> Note that the `Q4_0` model found [here](https://huggingface.co/unsloth/gpt-oss-20b-GGUF/blob/main/gpt-oss-20b-Q4_0.gguf) is a mixture of `Q4_0`, `Q8_0` and `MXFP4` and gives better performance than `MXFP4_MOE` quantization.

## CMake Options

The OpenCL backend has the following CMake options that control the behavior of the backend.


| CMake options                    | Default value | Description                                                                                 |
| -------------------------------- | ------------- | ------------------------------------------------------------------------------------------- |
| `GGML_OPENCL`                    | `OFF`         | Build stock `ggml-opencl` backend.                                                          |
| `GGML_OPENCL_KEPLER`             | `OFF`         | Build `ggml-opencl-kepler` (Kepler / OpenCL 1.2). **Not** combinable with `GGML_OPENCL=ON`. |
| `GGML_OPENCL_EMBED_KERNELS`      | `ON`          | Embed OpenCL kernels into the executable.                                                   |
| `GGML_OPENCL_USE_ADRENO_KERNELS` | `ON`          | Use kernels optimized for Adreno.                                                           |
| `GGML_OPENCL_USE_CLBLAST`        | `OFF`         | Use CLBlast for OpenCL matmul.                                                              |
| `GGML_OPENCL_LEGACY_NVIDIA`      | `OFF`         | Enable the OpenCL 1.2 / CLBlast compatibility path for old NVIDIA GPUs.                     |


### Legacy NVIDIA / Kepler Build

**Fork presets** use the dedicated backend (no dual OpenCL registration):

```powershell
cmake -B build-kepler -G Ninja `
  -DGGML_OPENCL=OFF `
  -DGGML_OPENCL_KEPLER=ON `
  -DGGML_OPENCL_LEGACY_NVIDIA=ON `
  -DGGML_OPENCL_USE_CLBLAST=ON `
  -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF `
  -DGGML_OPENCL_TARGET_VERSION=120
```

Optional stock OpenCL + legacy NVIDIA (upstream-style single backend name `GPUOpenCL`):

```powershell
cmake -B build-opencl-legacy -G Ninja `
  -DGGML_OPENCL=ON `
  -DGGML_OPENCL_LEGACY_NVIDIA=ON `
  -DGGML_OPENCL_USE_CLBLAST=ON `
  -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF `
  -DGGML_OPENCL_TARGET_VERSION=120
```

Notes:

- `GGML_OPENCL_LEGACY_NVIDIA=ON` forces the backend onto the OpenCL 1.2-safe CLBlast path.
- This mode intentionally disables the current subgroup-heavy OpenCL kernels.
- Unsupported operations continue on CPU through the normal multi-backend scheduler for nodes the GPU backend declines (`supports_op` false). The Kepler backend aims for **upstream OpenCL op parity** so full `-ngl` stays on GPU where the stock OpenCL backend would (see [opencl-kepler-op-parity-matrix.md](opencl-kepler-op-parity-matrix.md)).
- Large `MUL_MAT` weights (for example very wide embedding tables) may use chunked GPU dequant + multiple CLBlast GEMM calls so each staging buffer stays under `CL_DEVICE_MAX_MEM_ALLOC_SIZE`.
- The fork-specific GitHub Actions workflow uses the `fork-kepler-linux-release` and `fork-kepler-windows-release` presets from `CMakePresets.json`.

## Android

Ubuntu 22.04 is used for targeting Android. Make sure the following tools are accessible from command line,

- Git
- CMake 3.29
- Ninja
- Python3

### I. Setup Environment

1. **Install NDK**

```sh
cd ~
wget https://dl.google.com/android/repository/commandlinetools-linux-8512546_latest.zip && \
unzip commandlinetools-linux-8512546_latest.zip && \
mkdir -p ~/android-sdk/cmdline-tools && \
mv cmdline-tools latest && \
mv latest ~/android-sdk/cmdline-tools/ && \
rm -rf commandlinetools-linux-8512546_latest.zip

yes | ~/android-sdk/cmdline-tools/latest/bin/sdkmanager "ndk;26.3.11579264"
```

1. **Install OpenCL Headers and Library**

```sh
mkdir -p ~/dev/llm
cd ~/dev/llm

git clone https://github.com/KhronosGroup/OpenCL-Headers && \
cd OpenCL-Headers && \
cp -r CL ~/android-sdk/ndk/26.3.11579264/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include

cd ~/dev/llm

git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader && \
cd OpenCL-ICD-Loader && \
mkdir build_ndk26 && cd build_ndk26 && \
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$HOME/android-sdk/ndk/26.3.11579264/build/cmake/android.toolchain.cmake \
  -DOPENCL_ICD_LOADER_HEADERS_DIR=$HOME/android-sdk/ndk/26.3.11579264/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=24 \
  -DANDROID_STL=c++_shared && \
ninja && \
cp libOpenCL.so ~/android-sdk/ndk/26.3.11579264/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android
```

### II. Build llama.cpp

```sh
cd ~/dev/llm

git clone https://github.com/ggml-org/llama.cpp && \
cd llama.cpp && \
mkdir build-android && cd build-android

cmake .. -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$HOME/android-sdk/ndk/26.3.11579264/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_OPENCL=ON

ninja
```

## Windows 11 Arm64

A Snapdragon X Elite device with Windows 11 Arm64 is used. Make sure the following tools are accessible from command line,

- Git
- CMake 3.29
- Clang 19
- Ninja
- Visual Studio 2022
- Powershell 7
- Python

Visual Studio provides necessary headers and libraries although it is not directly used for building.
Alternatively, Visual Studio Build Tools can be installed instead of the full Visual Studio.

> Note that building using Visual Studio's cl compiler is not supported. Clang must be used. Clang depends on libraries provided by Visual Studio to work. Therefore, Visual Studio must be installed. Alternatively, Visual Studio Build Tools can be installed instead of the full Visual Studio.

Powershell 7 is used for the following commands.
If an older version of Powershell is used, these commands may not work as they are.

### I. Setup Environment

1. **Install OpenCL Headers and Library**

```powershell
mkdir -p ~/dev/llm

cd ~/dev/llm
git clone https://github.com/KhronosGroup/OpenCL-Headers && cd OpenCL-Headers
mkdir build && cd build
cmake .. -G Ninja `
  -DBUILD_TESTING=OFF `
  -DOPENCL_HEADERS_BUILD_TESTING=OFF `
  -DOPENCL_HEADERS_BUILD_CXX_TESTS=OFF `
  -DCMAKE_INSTALL_PREFIX="$HOME/dev/llm/opencl"
cmake --build . --target install

cd ~/dev/llm
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader && cd OpenCL-ICD-Loader
mkdir build && cd build
cmake .. -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_PREFIX_PATH="$HOME/dev/llm/opencl" `
  -DCMAKE_INSTALL_PREFIX="$HOME/dev/llm/opencl"
cmake --build . --target install
```

### II. Build llama.cpp

```powershell

mkdir -p ~/dev/llm
cd ~/dev/llm

git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
mkdir build && cd build

cmake .. -G Ninja `
  -DCMAKE_TOOLCHAIN_FILE="$HOME/dev/llm/llama.cpp/cmake/arm64-windows-llvm.cmake" `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_PREFIX_PATH="$HOME/dev/llm/opencl" `
  -DBUILD_SHARED_LIBS=OFF `
  -DGGML_OPENCL=ON
ninja
```

## Linux

The two steps just above also apply to Linux. When building for linux, the commands are mostly the same as those for PowerShell on Windows, but in the second step they do not have the `-DCMAKE_TOOLCHAIN_FILE` parameter, and then in both steps the backticks are replaced with back slashes.

If not installed already, install Git, CMake, Clang, Ninja and Python, then run in the terminal the following:

### I. Setup Environment

1. **Install OpenCL Headers and Library**

```bash
mkdir -p ~/dev/llm

cd ~/dev/llm
git clone https://github.com/KhronosGroup/OpenCL-Headers && cd OpenCL-Headers
mkdir build && cd build
cmake .. -G Ninja \
  -DBUILD_TESTING=OFF \
  -DOPENCL_HEADERS_BUILD_TESTING=OFF \
  -DOPENCL_HEADERS_BUILD_CXX_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX="$HOME/dev/llm/opencl"
cmake --build . --target install

cd ~/dev/llm
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader && cd OpenCL-ICD-Loader
mkdir build && cd build
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/dev/llm/opencl" \
  -DCMAKE_INSTALL_PREFIX="$HOME/dev/llm/opencl"
cmake --build . --target install
```

### II. Build llama.cpp

```bash
mkdir -p ~/dev/llm
cd ~/dev/llm

git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
mkdir build && cd build

cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/dev/llm/opencl" \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_OPENCL=ON
ninja
```

## Known Issues

- Flash attention does not always improve performance.
- Currently OpenCL backend works on A6xx GPUs with recent drivers and compilers (usually found in IoT platforms).
However, it does not work on A6xx GPUs found in phones with old drivers and compilers.
- The legacy NVIDIA / Kepler mode is intentionally conservative and currently offloads only the validated `MUL_MAT` path.

## TODO

- Optimization for Q6_K
- Support and optimization for Q4_K
- Improve flash attention

