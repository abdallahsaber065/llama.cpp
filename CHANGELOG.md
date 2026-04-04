# Changelog

## Unreleased

- Added a legacy NVIDIA / Kepler OpenCL compatibility profile that accepts OpenCL 1.2 devices, routes validated `MUL_MAT` workloads through CLBlast, and falls back to CPU for unsupported operations.
- Replaced the upstream multi-backend build matrix with a fork-specific GitHub Actions workflow and CMake presets focused on the Kepler OpenCL / CLBlast path, and documented the fork maintenance process under `docs/`.
- Fixed the fork-specific Kepler OpenCL CI by restoring the missing Windows OpenCL SDK setup and resolving the C++ forward-declaration break in the legacy CLBlast path.
- Fixed additional Kepler OpenCL CI regressions by guarding OpenCL 2.x/3.0-only profiling and buffer-allocation APIs when building the forced OpenCL 1.2 compatibility target.
