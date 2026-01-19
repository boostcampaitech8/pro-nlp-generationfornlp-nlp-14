#!/bin/bash
set -e

# CUDA 12.2 및 llama.cpp 설치 스크립트
# 실행: sudo bash setup.sh

apt-get update
apt-get install -y libxml2 wget rsync

cd /data/ephemeral/home

# 쿠다 툴킷 다운
wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run

# 권한 설정
chmod +x cuda_12.2.2_535.104.05_linux.run

# 임시파일 및 cuda toolkit 디렉토리
mkdir -p /data/ephemeral/home/src /data/ephemeral/home/build /data/ephemeral/home/tmp

# 쿠다 툴킷 설치 (overlay/opt, 6.8gb)
sh ./cuda_12.2.2_535.104.05_linux.run \
  --silent --override \
  --toolkit --toolkitpath=/data/ephemeral/home/build/cuda-12.2 \
  --no-opengl-libs \
  --tmpdir=/data/ephemeral/home/tmp

rm -rf \
  /data/ephemeral/home/build/cuda-12.2/nsight-compute-2023.2.2 \
  /data/ephemeral/home/build/cuda-12.2/nsight-systems-2023.2.3 \
  /data/ephemeral/home/build/cuda-12.2/libnvvp \
  /data/ephemeral/home/build/cuda-12.2/compute-sanitizer \
  /data/ephemeral/home/build/cuda-12.2/nsightee_plugins \
  /data/ephemeral/home/build/cuda-12.2/gds \
  /data/ephemeral/home/build/cuda-12.2/gds-12.2 \
  /data/ephemeral/home/build/cuda-12.2/extras \
  /data/ephemeral/home/build/cuda-12.2/src \
  /data/ephemeral/home/build/cuda-12.2/tools \
  /data/ephemeral/home/build/cuda-12.2/nvml

mkdir -p /opt/cuda-12.2

rsync -aH --info=progress2 --remove-source-files \
  /data/ephemeral/home/build/cuda-12.2/ /opt/cuda-12.2/

apt-get install -y git make build-essential jq moreutils

# 환경 변수 설정
export CUDA_HOME=/opt/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export CUDACXX=$CUDA_HOME/bin/nvcc
export TMPDIR=/data/ephemeral/home/tmp

nvcc --version | head -n 3

cd /data/ephemeral/home/src
git clone https://github.com/ggerganov/llama.cpp

cmake -S /data/ephemeral/home/src/llama.cpp -B /data/ephemeral/home/build/llama.cpp \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCUDAToolkit_ROOT=$CUDA_HOME \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF

cmake --build /data/ephemeral/home/build/llama.cpp -j 6

rm -rf /data/ephemeral/home/src /data/ephemeral/home/build/cuda-12.2 cuda_12.2.2_535.104.05_linux.run

echo "Setup completed successfully!"
