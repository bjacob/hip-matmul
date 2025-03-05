#!/bin/bash

set -eux

rm -rf build
if [[ ! -d build ]]
then
  mkdir build
fi

executable=build/hip-matmul

# TODO: specify your own clang build. by default rocm uses clang-19 which is outdated.
#export HIP_CLANG_PATH=~/llvm/bin
hipcc matmul.hip -target x86_64-unknown-linux-gnu -std=c++20 -Wall -Wextra -O3 -o "${executable}" -save-temps=obj

"${executable}"
