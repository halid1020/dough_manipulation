#  Dough Manipulation (DiffSkill Integration)

This repository integrates the modern [`Actoris Harena`](https://github.com/halid1020/actoris_harena/tree/develop) robot control framework with the legacy [`DiffSkill`](https://github.com/Xingyu-Lin/DiffSkill/tree/main) dough physics simulator. 

Because `DiffSkill` was released in 2021 and relies on older C++ compilation standards, setting up the environment on modern hardware requires a very specific installation sequence. **Please ignore the README of the orginal DiffSkill and follow the below instructions exactly as written instead.**

---

## ⚠️ Hardware & CUDA Compatibility
The instructions below default to **CUDA 11.1**, which is required for Ampere architecture GPUs (e.g., NVIDIA RTX 3090 Ti, RTX 30-series, RTX A2000). 

> **Using a different GPU?**
> * **Turing/Pascal (e.g., RTX 20-series, GTX 10-series):** You can use CUDA 10.2. In Steps 3 and 4 below, replace `cu111` with `cu102`, and install `cudatoolkit-dev=10.2`.
> * **Ada Lovelace/Hopper (e.g., RTX 40-series):** Note that PyTorch 1.9.0 does not natively support CUDA 11.8+. You may encounter hardware compatibility issues running this legacy environment on ultra-modern GPUs.

## 📋 Prerequisites
* **OS:** Known working ubuntu version: Ubuntu 20.04.5 LTS
* **Python:** `3.8`
* **Package Manager:** Conda

---

## 🛠️ Installation & Usage Guide

### Target Directory Structure
Before starting, ensure you are working within a unified workspace folder. By the end of Step 5, your directory should look exactly like this:
```text
your_workspace/
├── actoris_harena/       <-- (Branch: py3.8)
├── dough_manipulation/   <-- (This repository)
└── DiffSkill/            <-- (Original physics repository)
```

### 1. Workspace Setup & Virtual Environment
First, clone the required repositories and set up the Python 3.8 environment. 

```bash
# 1. Clone actoris_harena and strictly checkout the py3.8 branch
git clone -b py3.8 https://github.com/halid1020/actoris_harena.git

# 2. Clone this repository (if you haven't already)
git clone <URL_TO_DOUGH_MANIPULATION_REPO>

# 3. Create and activate the conda environment
conda create -n dough-manip python=3.8 -y
conda activate dough-manip
```
*(Note: To completely wipe a broken environment in the future, run `conda deactivate` followed by `conda remove -n dough-manip --all -y`)*

### 2. Prepare Legacy Build Tools
We must temporarily downgrade Python's build tools. `gym==0.21.0` contains a metadata typo (`opencv-python>=3.`) that modern versions of `pip` strictly reject. Downgrading `pip` makes it forgiving enough to ignore the typo, while older `setuptools` ensures it compiles correctly.

```bash
pip install "pip<24.1" setuptools==65.5.0 wheel==0.38.4
```

### 3. Install Core Packages
Navigate to the `actoris_harena` directory and run the installation. The `--extra-index-url` ensures we pull the PyTorch binaries compiled for your specific CUDA version (defaulting here to `cu111` for RTX 30-series).

```bash
cd actoris_harena
pip install -e ".[diffskill]" --extra-index-url https://download.pytorch.org/whl/cu111
```

### 4. Install the JIT CUDA Compiler
PyKeOps dynamically compiles custom C++ math operations (Earth Mover's Distance) for your specific GPU. We must install a matching `nvcc` compiler directly into the conda environment so it can build the GPU binaries.

```bash
conda install -c conda-forge nvcc_linux-64=11.1 cudatoolkit-dev=11.1 -y

# Refresh the environment to register the new compiler paths
conda deactivate
conda activate dough-manip
```

### 5. Clone and Prepare DiffSkill
Clone the original DiffSkill repository so it sits parallel to your other folders, then source the setup script.

```bash
cd ..
git clone https://github.com/Xingyu-Lin/DiffSkill.git

# Navigate into this repository to link the python paths
cd dough_manipulation
. ./setup.sh
```

### 6. Download Datasets
Download and extract the required target dough states into the `DiffSkill` directory.

```bash
cd ../DiffSkill
gdown 11XZw-p2FX-yvoHMnc_yNO5x7iiLxwlwB
unzip datasets.zip 
```

### 7. Test the Environment
Run the random policy script to verify the physics engine works. 

> **Compilation Pause:** The very first time you run this, the script will pause on a `[pyKeOps] Compiling...` step for 30–60 seconds while it builds the custom C++ kernels for your GPU. Subsequent runs will execute instantly.

```bash
cd ../dough_manipulatoin
source ./setup.sh
cd ../DiffSkill
python scripts/random_env.py --env_name LiftSpread-v1
```

---

## Architecture & Dependency Rationale (The "Why")

To prevent dependency drift, the `[diffskill]` block inside `actoris_harena/pyproject.toml` is strictly version-locked. Here is the engineering rationale behind these constraints:

* **`torch==1.9.0` (with cu111):** * *Why 1.9.0?* `pykeops==1.5` requires C++14 headers to compile its custom loss functions. Modern PyTorch (2.x) demands C++17 headers, causing compilation crashes. PyTorch 1.9.0 acts as the perfect C++14 bridge. 
  * *Why cu111?* The original code used CUDA 10.2, but Ampere GPUs (RTX 30-series) hardware-lock anything below CUDA 11.1.
* **`taichi==0.7.26`:** DiffSkill uses a custom 3D renderer called `tina`. This renderer relies on aggressive Python memory hacks (e.g., `ti.Matrix.is_global`) that were removed in Taichi version `0.8.0+`.
* **`gym==0.21.0`:** Modern Gym (`0.26+`) removed the `env.seed()` method and altered the `env.step()` output signature (returning 5 variables instead of 4). We lock to `0.21.0` so DiffSkill's legacy scripts execute natively without source code modifications.
* **`kornia<=0.6.8`:** Kornia `0.6.9+` strictly requires PyTorch `>=1.9.1`. Since we are hard-locked to PyTorch `1.9.0`, we must cap Kornia to prevent `pip` resolution deadlocks.
* **`cmake==3.24.0`:** Modern CMake (`3.29+`) dropped backward compatibility for the older `FindCUDA` modules utilized by PyKeOps 1.5. Version 3.24.0 safely interprets the older `CMakeLists.txt` files.
* **`natsort`:** Required by DiffSkill to correctly sequence point-cloud target files, but omitted from their original requirements.