#### Requirements

On ubuntu 24.04 x86_64:

- sudo apt update
- sudo apt install -y wget clang software-properties-common python3-pip
- wget https://github.com/bazelbuild/bazelisk/releases/download/v1.26.0/bazelisk-linux-amd64
- chmod +x bazelisk-linux-amd64
- sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel

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

#### Build wheels artifacts

Just launch:

    ./build_wheels.sh

The results would be in `./dist` directory
