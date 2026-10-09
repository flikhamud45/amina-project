# I1 + I2. Running on the TAU student GPUs: lean clones, Triton, torch.compile

Plan items I1 and I2. Status: done (scripts in the server folder `logs/validation/sp2027/`).

## 1. Lean clones (disk quota)

The project volume has a per-user quota of about 20 GB (`quota -s`), and full clones of the repository
take 344 MB each, 289 MB of it the stored benchmark records (`code/artifacts/comparison/raw*`), which jobs do
not read. `sp_setup.sh <name>` makes a clone in my folder that:

* reads the shared repository read-only (`git clone --no-checkout --upload-pack="git -c safe.directory='*'
  upload-pack" file://...`), so it never writes to anyone else's files;
* uses a sparse checkout without `code/artifacts/comparison/raw*/` (66 MB);
* takes branch `sp2027` from a bundle uploaded from the laptop (`sp2027.bundle`, incremental over the
  server's `submission` branch), and links the shared `.venv`.

Finished clones of a session are deleted (their outputs are under `out/` and copied to the laptop).

## 2. Exact int8 GEMMs on the RTX 2080 Ti

`field.int8_ok("cuda")` is True on the 2080 Ti (sm_75, torch 2.5.1+cu121): `torch._int_mm` matches float64
on the extremes and at the `2^30` accumulation bound. A1 and A2 therefore run there.

## 3. Triton and torch.compile on the GPU nodes

The compute nodes have gcc but neither g++ nor the Python development headers. Triton (3.1.0, shipped with
torch) compiles a small C launcher with gcc and `Python.h` on first use, so it failed. Fix, without
installing anything system-wide:

* a copy of the login node's headers in my folder: `pyinclude/python3.12/` (with the architecture's
  `pyconfig.h` from `/usr/include/x86_64-linux-gnu/python3.12/` copied over the generic one);
* in the job's Python, before importing Triton: point `sysconfig.get_paths()["include"]` at that copy
  (`PVI_PY_INCLUDE`), see `tri.py`;
* `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` in my folder (compute nodes cannot see `$HOME`).

Checked by job 1004245: a Triton kernel and `torch.compile` on CUDA both run and give exact results.
`torch.compile` on the CPU still failed at that point (inductor needs g++): plan I4, below.

## 4. A C++ compiler on the compute nodes (I4)

Inductor's CPU backend and our native attention (`native_kernels.py`, B5) need a C++ compiler. Without root,
conda-forge's `gxx` 13.4 was installed with a static micromamba into my folder, `logs/validation/sp2027/tools/cxx`
(951 MB after deleting the package cache, within the 20 GB quota). A CPU job sets

```
export CXX=$V/tools/cxx/bin/x86_64-conda-linux-gnu-g++ PATH=$V/tools/cxx/bin:$PATH PVI_NATIVE_DIR=$V/.native
export TORCHINDUCTOR_CACHE_DIR=$V/.inductor
```

Checked: `torch.compile` on the CPU builds and gives exact results (1.3-1.6x on the derive at 2,048 tokens, after a
20-120 s compile); `native_kernels.available()` is True on the nodes and its exactness tests pass (165, job 1005185).
