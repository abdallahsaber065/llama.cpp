# Changelog

## 0.1.0

- Added a legacy NVIDIA / Kepler OpenCL compatibility profile that accepts OpenCL 1.2 devices, routes validated `MUL_MAT` workloads through CLBlast, and falls back to CPU for unsupported operations.
- Replaced the upstream multi-backend build matrix with a fork-specific GitHub Actions workflow and CMake presets focused on the Kepler OpenCL / CLBlast path, and documented the fork maintenance process under `docs/`.
- Fixed the fork-specific Kepler OpenCL CI by restoring the missing Windows OpenCL SDK setup and resolving the C++ forward-declaration break in the legacy CLBlast path.
- Fixed additional Kepler OpenCL CI regressions by guarding OpenCL 2.x/3.0-only profiling and buffer-allocation APIs when building the forced OpenCL 1.2 compatibility target.
- Added automated GitHub Release publishing for successful `master` builds when `CHANGELOG.md` contains a real version heading, using auto-generated `build-<run_number>` tags and attaching both platform artifacts.
- Fixed the legacy Kepler runtime scheduler mismatch by enabling OpenCL `SET_ROWS` support for KV-cache writes in the CLBlast compatibility path.
- Fixed legacy NVIDIA runtime startup on OpenCL 1.2 drivers without FP16 support by skipping FP16 `SET_ROWS` kernel compilation and keeping incompatible F16 KV-cache buffers on CPU.
- Replaced the old upstream-oriented `AGENTS.md` with fork-specific persistent guidance covering this fork's Kepler/OpenCL goal, key files, maintenance workflow, and mandatory changelog/docs updates after changes.
