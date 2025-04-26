#### Requirements

On ubuntu 24.04 x86_64:

- sudo apt update
- sudo apt install -y wget clang software-properties-common python3-pip
- wget https://github.com/bazelbuild/bazelisk/releases/download/v1.26.0/bazelisk-linux-amd64
- chmod +x bazelisk-linux-amd64
- sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
- curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
- bash Anaconda3-2024.10-1-Linux-x86_64.sh
- rm Anaconda3-2024.10-1-Linux-x86_64.sh
- ~/anaconda3/bin/conda init
- conda create --name JAX-060-CUDA-1251 python=3.12.8
- conda activate JAX-060-CUDA-1251
- conda install cuda-version=12.5
- conda install nvidia/label/cuda-12.5.0::cuda-nvcc
- conda install nvidia/label/cuda-12.5.0::libcublas-dev
- conda install nvidia/label/cuda-12.5.0::cuda-nvml-dev
- conda install -c nvidia cudnn

After repo clone:

- git submodule update --init

#### Overview

Current build process tested on `NV6ads_A10_v5` Azure VM instance.

Root directory of this repo contains Bazel configuration for XLA and JAX included as submodules.
JAX's `build.py` script is quite complex, here we have configuration reduced.

This repo is considered as a playground, but build process from source is quite heavy and slow.
So, build configuration cached for hermeticity and reproducibility only to build with CUDA on target machine.

Addtional Bazel configuration flags added in `.bazelrc`, e.g.:

    --disk_cache=~/bazel/disk_cache
    --repository_cache=~/bazel/repository_cache

This will help to cache intermediate artifacts and reduce rebuild times between VM restarts. 

Because JAX is included as a subdirectory and not in root, I had to fix some paths in build scripts.
Here is the fork and branch:

    https://github.com/gleyba/jax/tree/jax-v.0.6.0-gleb

The same can be done for XLA, just fork it on github and change origin for submodule:

    git submodule set-url xla https://github.com/%%github_username%%/xla.git

Then create a branch in `xla` subdirectory, commit changes and push

#### Build release wheels artifacts

Just launch:

    ./build_wheels.sh

The results would be in `./dist` directory

#### Playground

There is 'playground.py' file with JAX initialized to be ready to run from sources.
To launch an examples, simply do:

    bazel run //:playground

On the first run it will build XLA and JAX from sources. 
But subsequent runs will reuse already built artifacts and run python only.

If you want to apply changes to XLA or JAX codebase, after doing so, it is easy to test changes by re-runing playground. 
Bazel will incrementally build only changed parts of code.
