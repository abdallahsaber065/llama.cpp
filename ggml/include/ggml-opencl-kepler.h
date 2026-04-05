#ifndef GGML_OPENCL_KEPLER_H
#define GGML_OPENCL_KEPLER_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Dedicated OpenCL 1.2 + CLBlast backend for NVIDIA Kepler-class GPUs (e.g. Quadro K3100M).
// Built from the same op surface as ggml-opencl with legacy NVIDIA constraints; mutually exclusive
// with GGML_OPENCL in CMake (only one OpenCL backend per binary).

GGML_BACKEND_API ggml_backend_t     ggml_backend_opencl_kepler_init(void);
GGML_BACKEND_API bool               ggml_backend_is_opencl_kepler(ggml_backend_t backend);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_opencl_kepler_reg(void);

#ifdef __cplusplus
}
#endif

#endif // GGML_OPENCL_KEPLER_H
