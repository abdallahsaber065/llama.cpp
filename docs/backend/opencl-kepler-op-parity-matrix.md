# OpenCL-Kepler upstream op parity checklist (model-blind)

**Goal:** Match the **same** `GGML_OP` surface that upstream `ggml-opencl` advertises via `supports_op` / dispatch, adapted for **OpenCL 1.2**, **no FP16 extension**, **no subgroups**, and **CLBlast** where used. The dedicated backend registers as **`OpenCL-Kepler`** (`GPUOpenCLKepler`).

**Policy:** Only return `true` from `supports_op` when the node can run **entirely on the GPU** on this backend. If the scheduler assigns a node here, execution must not silently fall back to CPU for that op (parity with upstream OpenCL expectations; VRAM/OOM is separate).

**Source of truth (refresh when syncing `main`):**

- `ggml_opencl_supports_op_standard` — primary dtype/shape matrix: `ggml/src/ggml-opencl/ggml-opencl.cpp` (search for `ggml_opencl_supports_op_standard`).
- `ggml_opencl_supports_op` — legacy NVIDIA branch: special-cases `MUL_MAT` via `ggml_opencl_can_mul_mat_legacy`, then defers other ops to `ggml_opencl_supports_op_standard`.
- `ggml_cl_compute_forward` — must implement every op path that `supports_op` can return true for (same file; large switch).

The Kepler backend (`ggml/src/ggml-opencl-kepler/ggml-opencl-kepler.cpp`) is a **fork** of that implementation with a distinct registry GUID/name, NVIDIA-only probe when `GGML_OPENCL_KEPLER_BUILD` is defined, device tag **`[opencl_kepler]`** (plus **`[opencl_legacy_no_fp16]`** when applicable), and **no** `GGML_OPENCL_SOA_Q`.

## Checklist by `GGML_OP` (high level)

Use this table as a working backlog; tick rows when Kepler `supports_op` and `graph_compute` paths are verified together.

| GGML_OP | Upstream OpenCL (`supports_op_standard` + legacy) | Kepler backend notes |
|---------|-----------------------------------------------------|----------------------|
| `NONE`, `RESHAPE`, `VIEW`, `PERMUTE`, `TRANSPOSE` | Yes (legacy branch explicit) | Same |
| `GET_ROWS` | F32/F16 rows; Q4_0 (unless SOA_Q) | Same; no SOA_Q in Kepler target |
| `SET_ROWS` | F32 src0; dst F16/F32; indices I32/I64 | Same |
| `CPY`, `DUP`, `CONT` | F32/F16/I32 combinations per switch | Same |
| `SET` | F32/I32 equality constraints | Same |
| `SCALE` | F32 contiguous | Same |
| `ADD`, `MUL`, `DIV`, `SUB` | F32/F16 rules | Same |
| `ADD_ID` | F32 | Same |
| `SQR`, `SQRT` | F32/F16 contiguous | Same |
| `UNARY` | Subset (GELU, SILU, …) | Same |
| `GLU` | Several `GGML_GLU_OP_*` | Same |
| `TRI`, `FILL`, `CLAMP` | F32 | Same |
| `SOFT_MAX`, `NORM` | Broad | Same |
| `RMS_NORM` | `ne[0] % 4`, contiguous rows | Same |
| `L2_NORM` | Contiguous rows | Same |
| `REPEAT`, `PAD`, `UPSCALE` | F32 / mode constraints | Same |
| `CONV_2D` | F16/F32 combinations | Same |
| `SSM_CONV` | F32 | Same |
| `CONCAT`, `TIMESTEP_EMBEDDING` | F32 | Same |
| `GROUP_NORM` | Contiguous | Same |
| `MUL_MAT` | Quant + F16/F32 + legacy `can_mul_mat_legacy` | CLBlast + tiling; native GEMM optional |
| `MUL_MAT_ID` | Q4_0, Q8_0, MXFP4 + F32 | Same matrix |
| `DIAG`, `DIAG_MASK_INF` | Constraints | Same |
| `ROPE` | Mrope / vision / NeoX rules | Same |
| `SOLVE_TRI` | F32 | Same |
| `IM2COL` | Yes | Same |
| `ARGSORT` | F32 + workgroup limit | Same |
| `SUM_ROWS`, `CUMSUM`, `MEAN` | F32 | Same |
| `FLASH_ATTN_EXT` | Fixed (dk,dv) list + dtype combos | Same |

## Validation (manual)

On hardware:

1. Build with `GGML_OPENCL_KEPLER=ON`, `GGML_OPENCL=OFF` (fork presets).
2. Run short generations with **full** `-ngl` on several GGUF families (dense LM, optional MoE if you use them).
3. Compare logits or decoded text to a **CPU reference** run on the same prompt (short max_tokens).
4. If a node is wrongly placed on Kepler, enable `GGML_OPENCL_LEGACY_OP_INVENTORY=1` on the legacy OpenCL build or inspect scheduler logs as upstream documents.

CI builds this target **compile-only**; GPU parity is owner-validated.
