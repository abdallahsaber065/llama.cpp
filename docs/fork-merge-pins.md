# Fork merge pins (upstream sync)

When you merge or rebase **upstream llama.cpp** into this fork, these paths are the ones you usually want to **keep fork-specific** or **resolve carefully**. The repo root **`.gitattributes`** sets `merge=ours` on a small set of files so Git prefers **your branch’s version** during a merge conflict resolution for those paths (you still must run `git merge`; on conflict, the fork copy wins for pinned files).

## Always review after pulling upstream

| Area | Paths | Why |
|------|--------|-----|
| Kepler CI / releases | `.github/workflows/fork-kepler-opencl-build.yml` | Only active workflow; defines Linux/Windows Kepler builds and automatic releases. |
| Disabled upstream automation | `.github/workflows/*.yml.disabled` | Upstream workflows are turned off in this fork; upstream may add new `.yml` files you may want to disable the same way. |
| Build presets | `CMakePresets.json`, top-level `CMakeLists.txt` (if fork-touched) | Fork presets (`fork-kepler-*`) and options must stay. |
| OpenCL Kepler backend | `ggml/src/ggml-opencl-kepler/**`, `ggml/include/ggml-opencl-kepler.h` | Fork backend implementation. |
| OpenCL integration | `ggml/src/ggml-backend-reg.cpp`, `ggml/src/CMakeLists.txt`, `ggml/CMakeLists.txt` | Registration and CMake wiring for `GGML_OPENCL_KEPLER`. |
| Runtime fork hooks | `src/llama-context.cpp`, `src/llama-kv-cache.cpp` (if fork-touched) | Device tags / KV behavior for Kepler. |
| Docs / policy | `AGENTS.md`, `docs/kepler-opencl-fork.md`, `docs/backend/OPENCL.md`, `docs/backend/opencl-kepler-*.md` | Fork documentation. |
| Changelog | `CHANGELOG.md` | Append upstream notes or keep a fork section; `merge=ours` is **not** applied so you can merge text—resolve by hand. |

## GitHub Actions in this fork

- **Enabled:** `.github/workflows/fork-kepler-opencl-build.yml` only (Linux + Windows, OpenCL Kepler preset, **automatic GitHub Release** on every successful `master` push with a unique tag `kepler-opencl-r<run>-a<attempt>-<sha7>`).
- **Disabled:** all other former workflows live as `.github/workflows/*.yml.disabled`. After syncing upstream, if new `*.yml` appear, rename them to `*.yml.disabled` if you do not want them to run.

## Refreshing the op / device matrices

After large upstream OpenCL changes, refresh:

- `docs/backend/opencl-kepler-op-parity-matrix.md`
- `docs/backend/opencl-kepler-device-matrix.md`

against upstream `ggml/src/ggml-opencl/ggml-opencl.cpp` and `ggml/include/ggml.h`.
