# Base image matches your required CUDA 11.1 and Ubuntu 20.04 OS
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

# Prevent interactive timezone prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Install system prerequisites
RUN apt-get update && apt-get install -y \
    git wget curl unzip build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Create the unified workspace directory
WORKDIR /workspace

# Clone the static repositories (These will live permanently inside the container)
RUN git clone -b py3.8 https://github.com/halid1020/actoris_harena.git && \
    git clone https://github.com/Xingyu-Lin/DiffSkill.git

# Accept Conda Terms of Service for non-interactive builds
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create the Conda environment and install legacy build tools
RUN conda create -n dough-manip python=3.8 -y && \
    conda run -n dough-manip pip install "pip<24.1" setuptools==65.5.0 wheel==0.38.4
    
# Install actoris_harena with the strict CUDA 11.1 wheels
RUN cd actoris_harena && \
    conda run -n dough-manip pip install -e ".[diffskill]" --extra-index-url https://download.pytorch.org/whl/cu111

# Install the custom GPU compilers via conda-forge
RUN conda run -n dough-manip conda install -c conda-forge nvcc_linux-64=11.1 cudatoolkit-dev=11.1 -y

# Download and unzip the target datasets into DiffSkill
RUN cd DiffSkill && \
    conda run -n dough-manip pip install gdown && \
    conda run -n dough-manip gdown 11XZw-p2FX-yvoHMnc_yNO5x7iiLxwlwB && \
    unzip datasets.zip && rm datasets.zip

# Setup bashrc to auto-activate the conda environment on login
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate dough-manip" >> ~/.bashrc

# Export Docker-native Environment Variables (Replaces the need for setup.sh paths)
ENV PYTHONPATH="/workspace/dough_manipulation:/workspace/DiffSkill:/workspace/DiffSkill/taichi_three:/workspace/DiffSkill/PointFlow:/workspace/DiffSkill/setvae:${PYTHONPATH}"
ENV PATH="/workspace/DiffSkill/taichi_three:${PATH}"

# Set working directory to the folder we will mount
WORKDIR /workspace/dough_manipulation

# Keep the container running in the background
CMD ["tail", "-f", "/dev/null"]