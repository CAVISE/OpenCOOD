# Installation

## System and hardware requirements

- The CAVISE integration is tested under Ubuntu 24.04.
- Python 3.12 or newer is required by this fork.
- A CUDA-capable GPU with at least 6 GB of memory is recommended.
- Around 100 GB of free disk space is recommended for datasets.

## Source installation

Clone OpenCOOD next to OpenCDA, install its standalone dependencies, and then
install the source tree in editable mode:

```sh
git clone https://github.com/CAVISE/opencood.git
cd opencood
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps --editable .
```

The requirements file intentionally includes dependencies that may also be
installed by OpenCDA. OpenCOOD remains usable as a standalone source checkout.
The `--no-deps` option on the second command avoids resolving those packages a
second time; the first command has already installed them using the CUDA wheel
indexes declared in `requirements.txt`.

## Dependencies for FPV-RCNN

OpenCOOD owns the CUDA extensions used by FPV-RCNN. Install the smaller build
dependency set, configure an explicit compute capability, and build from this
repository's root:

```sh
python -m pip install -r requirements-cuda.txt
cmake --preset cuda -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build --preset cuda
cmake --install build/cuda --prefix build/cuda-install
```

The CAVISE `opencda-cuda` and full `opencda` Docker targets run the same
OpenCOOD-owned CMake build in an isolated builder stage. At container startup,
OpenCOOD's entrypoint synchronizes those extensions into the mounted source
tree. Rebuild one of those targets after changing a `.cpp` or `.cu` source
under `opencood/pcdet_utils`.
