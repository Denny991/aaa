#!/bin/bash
# Decode 节点部署脚本 - PD 分离架构（同容器双卡版本）
# 同一容器内，GPU 0 运行 Prefill，GPU 1 运行 Decode
# Proxy 独立运行在 100.100.254.204:30001
# 使用: bash run_d_local_2gpu.sh

LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64
PATH=/usr/local/nvidia/bin/:/usr/local/nvidia/bin/:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64
export LIBRARY_PATH=/usr/lib/:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64
export LD_LIBRARY_PATH=/usr/lib/:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64

rm -rf /tmp/torchinductor_*
export FLASHINFER_DISABLE_VERSION_CHECK=1


export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
# 容器内 IB 设备不可用，禁用 IB 防止 ncclSend/ncclRecv 挂死
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
# 同容器双卡：无需禁用 GPU P2P，KV transfer 走 NVLink/PCIe
# export NCCL_P2P_DISABLE=1
# 跨进程 NCCL 必须关闭 cuMem，否则 CommInit(CUMEM=1) 与 Send/Recv 状态不一致
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_DEBUG=WARN
# 同容器双卡：Decode 使用 GPU 1
export CUDA_VISIBLE_DEVICES=1

# 集群配置
PROC_PER_NODE=1
NODE_RANK=1

echo "=========================================="
echo "Decode (D) - PD 分离部署 [同容器双卡，GPU 1]"
echo "=========================================="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "NODE_RANK: $NODE_RANK"
echo "PROC_PER_NODE: $PROC_PER_NODE"
echo "Proxy: 100.100.254.204:30001"
echo "=========================================="

# ========== vLLM Decode 实例启动 ==========
echo "[vLLM] Starting Decode instance on port 8200..."

vllm serve /mnt/shared-storage-user/ailab-sys/caikun/models/Qwen/Qwen3-4B --served-model-name Qwen3-4B \
    --host 0.0.0.0 \
    --port 8200 \
    --tensor-parallel-size ${PROC_PER_NODE} \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    --distributed-executor-backend mp \
    --kv-transfer-config '{
        "kv_connector": "P2pNcclConnector",
        "kv_role": "kv_consumer",
        "kv_buffer_size": 2e9,
        "kv_port": 22001,
        "kv_connector_extra_config": {
            "proxy_ip": "100.100.254.204",
            "proxy_port": "30001",
            "http_port": 8200
        }
    }' \
    --no-enable-prefix-caching

# --disable-custom-all-reduce \
# --no-enable-chunked-prefill \
# --enforce-eager \


# mv /mnt/shared-storage-user/ailab-sys/liutong/mycodes/dp_vllm_2_local/p2p_nccl_engine.py /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py && mv /mnt/shared-storage-user/ailab-sys/liutong/mycodes/dp_vllm_2_local/p2p_nccl_connector.py /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py


# mv /mnt/shared-storage-user/ailab-sys/liutong/mycodes/dp_vllm_2_local/p2p_nccl_engine.py /vllm-workspace/vllm/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py && mv /mnt/shared-storage-user/ailab-sys/liutong/mycodes/dp_vllm_2_local/p2p_nccl_connector.py /vllm-workspace/vllm/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py \

