
# Dough Manipulation (DiffSkill Integration)

This repository integrates the modern [`Actoris Harena`](https://github.com/halid1020/actoris_harena/tree/develop) robot control framework with the legacy [`DiffSkill`](https://github.com/Xingyu-Lin/DiffSkill/tree/main) dough physics simulator. 

Because `DiffSkill` was released in 2021 and relies on older C++ compilation standards, setting up the environment on modern hardware requires a very specific installation sequence. **Please ignore the README of the original DiffSkill and follow the below instructions exactly as written instead.**

---

## Table of Contents
1. [Hardware & CUDA Compatibility](#hardware--cuda-compatibility)
2. [Installation Method 1: Docker & VS Code (Recommended)](#installation-method-1-docker--vs-code-recommended)
3. [Installation Method 2: Local Conda Environment](#installation-method-2-local-conda-environment)
4. [Architecture & Dependency Rationale (The "Why")](#architecture--dependency-rationale-the-why)

---

## Hardware & CUDA Compatibility
The instructions below default to **CUDA 11.1**, which is required for Ampere architecture GPUs (e.g., NVIDIA RTX 3090 Ti, RTX 30-series, RTX A2000). 

> **Using a different GPU?**
> * **Turing/Pascal (e.g., RTX 20-series, GTX 10-series):** You can use CUDA 10.2. In the local installation steps, replace `cu111` with `cu102`, and install `cudatoolkit-dev=10.2`.
> * **Ada Lovelace/Hopper (e.g., RTX 40-series):** Note that PyTorch 1.9.0 does not natively support CUDA 11.8+. You may encounter hardware compatibility issues running this legacy environment on ultra-modern GPUs.

---


## Installation Method 1: Docker & VS Code (Recommended)

This repository includes a `Dockerfile` and `docker-compose.yml` that fully automates the legacy C++ and CUDA 11.1 environment setup. It mounts your local `dough_manipulation` directory into the container, allowing you to edit files on your host machine while running the code inside the isolated container.

### 1. Docker Prerequisites

**For Linux Hosts:**
* [Docker](https://docs.docker.com/engine/install/ubuntu/) and Docker Compose.
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (required for GPU access).

**For Windows Hosts:**
* Install **WSL 2** (Windows Subsystem for Linux).
* Install [Docker Desktop](https://docs.docker.com/desktop/install/windows/) and ensure the **WSL 2 backend** is enabled in the settings.
* Ensure your standard NVIDIA Windows Display Drivers are up to date. *(Note: You do **not** need to install a separate NVIDIA Container Toolkit on Windows; GPU passthrough is handled natively by WSL 2).*
* ⚠️ **Crucial Git Setting:** Windows uses `CRLF` line endings by default, which will crash Linux bash scripts inside the container. Before cloning any repositories, open your Windows terminal and run `git config --global core.autocrlf false` to force `LF` line endings.

**For All Users:**
* **VS Code** with the "Dev Containers" extension installed.

### 2. Build and Start the Container
Navigate to this repository on your host machine and start the container. This will download the datasets, compile the wheels, and set up the environment (this takes 10-15 minutes the first time).

```bash
cd dough_manipulation
docker compose up -d --build
```
*(Note: If you encounter network/DNS errors during build, restart your Docker daemon. On Linux, run `sudo systemctl restart docker`. On Windows, click the Docker Desktop system tray icon and select "Restart". If that fails, add `"dns": ["8.8.8.8"]` to your Docker Engine JSON configuration).*

### 3. Connect your Editor
1. Open VS Code.
2. Click the green `><` button in the bottom left corner (or open the Command Palette `Ctrl+Shift+P`).
3. Select **"Dev Containers: Attach to Running Container..."**
4. Select `dough_manip_container` from the list.

### 4. Test the Environment
Open the integrated terminal in your attached VS Code window (`Ctrl+~`). The terminal will open in `/workspace/dough_manipulation` with the `dough-manip` conda environment already activated and all paths exported.

Create the dataset symlink and run the test script:

```bash
# Create the soft link for the dataset
ln -sfn ../DiffSkill/datasets datasets

# Run the random policy script to verify the physics engine works
cd ../DiffSkill
python scripts/random_env.py --env_name LiftSpread-v1
```
*(Note: The very first time you run this, the script will pause on a `[pyKeOps] Compiling...` step for 30–60 seconds. Subsequent runs will execute instantly.)*

---

## Installation Method 2: Local Conda Environment

### Prerequisites
* **OS:** Ubuntu 20.04.5 LTS (Known working version)
* **Python:** `3.8`

---

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
git clone -b py3.8 [https://github.com/halid1020/actoris_harena.git](https://github.com/halid1020/actoris_harena.git)

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
pip install -e ".[diffskill]" --extra-index-url [https://download.pytorch.org/whl/cu111](https://download.pytorch.org/whl/cu111)
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
git clone [https://github.com/Xingyu-Lin/DiffSkill.git](https://github.com/Xingyu-Lin/DiffSkill.git)

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

```bash
cd ../dough_manipulation
source ./setup.sh
cd ../DiffSkill
python scripts/random_env.py --env_name LiftSpread-v1
```

---

## Architecture & Dependency Rationale (The "Why")

To prevent dependency drift, the `[diffskill]` block inside `actoris_harena/pyproject.toml` is strictly version-locked. Here is the engineering rationale behind these constraints:

* **`torch==1.9.0` (with cu111):** 
  * *Why 1.9.0?* `pykeops==1.5` requires C++14 headers to compile its custom loss functions. Modern PyTorch (2.x) demands C++17 headers, causing compilation crashes. PyTorch 1.9.0 acts as the perfect C++14 bridge. 
  * *Why cu111?* The original code used CUDA 10.2, but Ampere GPUs (RTX 30-series) hardware-lock anything below CUDA 11.1.
* **`taichi==0.7.26`:** DiffSkill uses a custom 3D renderer called `tina`. This renderer relies on aggressive Python memory hacks (e.g., `ti.Matrix.is_global`) that were removed in Taichi version `0.8.0+`.
* **`gym==0.21.0`:** Modern Gym (`0.26+`) removed the `env.seed()` method and altered the `env.step()` output signature (returning 5 variables instead of 4). We lock to `0.21.0` so DiffSkill's legacy scripts execute natively without source code modifications.
* **`kornia<=0.6.8`:** Kornia `0.6.9+` strictly requires PyTorch `>=1.9.1`. Since we are hard-locked to PyTorch `1.9.0`, we must cap Kornia to prevent `pip` resolution deadlocks.
* **`cmake==3.24.0`:** Modern CMake (`3.29+`) dropped backward compatibility for the older `FindCUDA` modules utilized by PyKeOps 1.5. Version 3.24.0 safely interprets the older `CMakeLists.txt` files.
* **`natsort`:** Required by DiffSkill to correctly sequence point-cloud target files, but omitted from their original requirements.