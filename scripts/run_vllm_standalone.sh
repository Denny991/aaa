#!/bin/bash
# 单节点 vLLM 部署脚本（非PD分离）
# 使用: bash run_vllm_standalone.sh

LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64
PATH=/usr/local/nvidia/bin/:/usr/local/nvidia/bin/:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64
export LIBRARY_PATH=/usr/lib/:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64
export LD_LIBRARY_PATH=/usr/lib/:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64

rm -rf /tmp/torchinductor_*
export FLASHINFER_DISABLE_VERSION_CHECK=1

# 超时配置
export VLLM_RPC_TIMEOUT=600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200

# 禁用 custom all-reduce
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1

# 网络配置
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_DEBUG=INFO
export VLLM_LOGGING_LEVEL=DEBUG
export GLOO_SOCKET_IFNAME=eth0

# 集群配置
TENSOR_PARALLEL_SIZE=1

echo "=========================================="
echo "vLLM 单节点部署"
echo "=========================================="
echo "TENSOR_PARALLEL_SIZE: $TENSOR_PARALLEL_SIZE"
echo "=========================================="

# ========== vLLM 启动 ==========
echo "[vLLM] Starting server on port 8000..."

vllm serve /mnt/shared-storage-user/ailab-sys/caikun/models/Qwen/Qwen3-4B \
    --served-model-name Qwen3-4B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    --distributed-executor-backend ray \
    --disable-custom-all-reduce
