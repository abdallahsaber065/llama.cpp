# Changelog

## Unreleased

- Added a legacy NVIDIA / Kepler OpenCL compatibility profile that accepts OpenCL 1.2 devices, routes validated `MUL_MAT` workloads through CLBlast, and falls back to CPU for unsupported operations.
- Replaced the upstream multi-backend build matrix with a fork-specific GitHub Actions workflow and CMake presets focused on the Kepler OpenCL / CLBlast path, and documented the fork maintenance process under `docs/`.
