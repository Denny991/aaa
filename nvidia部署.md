# NVIDIA + vLLM + DLRouter 部署文档（含 PD 分离）

本文整理了在 NVIDIA GPU 环境下使用官方 vLLM 镜像部署节点并接入 DLRouter 的完整流程，包含标准 Hybrid 模式和 PD 分离（Prefill-Decode Disaggregation）模式。

---

## 1. 前置条件

- 宿主机已安装 Docker，且 `nvidia-container-toolkit` 已配置（即 `docker run --gpus` 可用）。
- 模型目录已就绪（示例使用 `/datapool/models/Qwen3-32B`）。
- DLRouter 代码可运行（`pip install -e .` 或通过 conda 环境）。
- PD 分离模式额外要求：节点间网络互通（IB/RDMA 推荐 NixlConnector，TCP 可用 LMCacheConnector）。

---

## 2. 镜像说明

- 官方 vLLM 镜像：`vllm/vllm-openai:latest`（或指定版本如 `v0.6.6`）
- vLLM PD 分离需要 **v0.6.0+**，建议使用最新稳定版本。

---

## 3. 标准模式（Hybrid）部署

### 3.1 启动 vLLM 容器

```bash
docker run -d \
  --gpus all \
  --ipc=host \
  --network=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  -v /datapool/models:/models \
  --name vllm_node1 \
  vllm/vllm-openai:latest \
  --model /models/Qwen3-32B \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16 \
  --port 8001
```

本地验证：

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3-32B",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

### 3.2 启动 DLRouter（Hybrid 模式）

```bash
dlrouter \
  --backend vllm \
  --server_name 0.0.0.0 \
  --server_port 8000 \
  --routing_strategy min_observed_latency \
  --serving_strategy hybrid \
  --log_level INFO
```

### 3.3 注册节点并验证

```bash
# 注册节点
curl -X POST http://localhost:8000/nodes/add \
  -H "Content-Type: application/json" \
  -d '{"url": "http://127.0.0.1:8001"}'

# 查看节点状态
curl http://localhost:8000/nodes/status

# 通过 DLRouter 转发请求
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3-32B",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

---

## 4. PD 分离模式（DistServe）部署

### 4.1 机制说明

```
用户请求
    │
    ▼
DLRouter (serving_strategy=distserve)
    │
    ├─── [Phase 1] POST → Prefill 节点 (max_tokens=1)
    │         │  vLLM 完成 prefill，KV 经 connector 传输到 D 节点
    │         └─ 返回首个 token
    │
    └─── [Phase 2] POST → Decode 节点 (追加首 token，继续生成)
              │  D 节点从 connector 加载 KV（跳过重算）
              └─ 返回完整生成结果
```

P 节点和 D 节点通过 `--kv-transfer-config` 配置 KV 传输 connector，DLRouter 只负责请求的两阶段分发。

### 4.2 Connector 选择

| Connector | 适用场景 | 是否支持跨机 | 说明 |
|---|---|---|---|
| `PyNcclConnector` | **同机**多进程（P/D 在同节点不同 GPU）| 否 | 使用 NCCL 共享内存，最简单；跨机不可用 |
| `NixlConnector` | **跨机**，有 IB 或 RoCE | 是 | RDMA 直传，延迟最低；需配置 `kv_ip`/`kv_port` |
| `MooncakeConnector` | **跨机**，阿里云/月之暗面环境 | 是 | 基于 Mooncake 传输引擎 |
| `LMCacheConnector` | **跨机**，普通以太网（无 RDMA）| 是 | 经 Redis 中转，兼容性最好，延迟较高 |

### 4.3 启动 DLRouter（DistServe 模式）

**DLRouter 需要在 P/D 节点注册之前先启动**：

```bash
dlrouter \
  --backend vllm \
  --server_name 0.0.0.0 \
  --server_port 8000 \
  --routing_strategy min_observed_latency \
  --serving_strategy distserve \
  --log_level INFO
```

---

### 4.4 同机 PD 分离（PyNcclConnector，P/D 在同一台 8 卡机器）

#### P 节点（GPU 0-3，端口 8001）

```bash
docker run -d \
  --gpus "device=0,1,2,3" \
  --ipc=host \
  --network=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  -v /datapool/models:/models \
  --name vllm_prefill \
  vllm/vllm-openai:latest \
  --model /models/Qwen3-32B \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16 \
  --port 8001 \
  --kv-transfer-config \
  '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}'
```

#### D 节点（GPU 4-7，端口 8002）

```bash
docker run -d \
  --gpus "device=4,5,6,7" \
  --ipc=host \
  --network=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  -v /datapool/models:/models \
  --name vllm_decode \
  vllm/vllm-openai:latest \
  --model /models/Qwen3-32B \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16 \
  --port 8002 \
  --kv-transfer-config \
  '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}'
```

> **注意**：`kv_parallel_size` 为该通信组内 P+D 总实例数。`kv_rank` 必须唯一且覆盖 `[0, kv_parallel_size)`。

#### 注册节点并验证

```bash
# 注册 P 节点
curl -X POST http://localhost:8000/nodes/add \
  -H "Content-Type: application/json" \
  -d '{"url": "http://127.0.0.1:8001", "role": "PREFILL"}'

# 注册 D 节点
curl -X POST http://localhost:8000/nodes/add \
  -H "Content-Type: application/json" \
  -d '{"url": "http://127.0.0.1:8002", "role": "DECODE"}'

# 查看状态
curl http://localhost:8000/nodes/status

# 发送请求验证
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3-32B",
    "messages": [{"role": "user", "content": "请写一首关于夏天的诗"}],
    "stream": true
  }'
```

---

### 4.5 跨机部署方案一：NixlConnector（有 IB/RoCE）

NixlConnector 使用 NVIDIA NIXL 库进行跨机 RDMA 直传，是跨机 PD 分离的最优方案。
**前提**：两台机器之间有 InfiniBand 或 RoCE 网络，容器内 `ibv_devices` 可见。

#### P 节点（机器 A，IP=10.201.6.4，端口 8001）

```bash
docker run -d \
  --gpus all \
  --ipc=host \
  --network=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --device=/dev/infiniband \
  -v /datapool/models:/models \
  --name vllm_prefill \
  vllm/vllm-openai:latest \
  --model /models/Qwen3-32B \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16 \
  --port 8001 \
  --kv-transfer-config \
  '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_connector_extra_config":{"kv_ip":"10.201.6.4","kv_port":14579}}'
```

#### D 节点（机器 B，IP=10.201.6.5，端口 8001）

```bash
docker run -d \
  --gpus all \
  --ipc=host \
  --network=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --device=/dev/infiniband \
  -v /datapool/models:/models \
  --name vllm_decode \
  vllm/vllm-openai:latest \
  --model /models/Qwen3-32B \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16 \
  --port 8001 \
  --kv-transfer-config \
  '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_connector_extra_config":{"kv_ip":"10.201.6.5","kv_port":14579}}'
```

> **说明**：
> - `kv_ip`：本机对外网卡 IP（NIXL agent 监听地址），跨机时**必须指定**，否则默认绑定 localhost，对端无法连接。
> - `kv_port`：NIXL agent 监听端口，默认 14579，确保防火墙放行。
> - `--device=/dev/infiniband`：将宿主机 IB HCA 透传进容器；使用 RoCE 则无需此参数，`--network=host` 即可。
> - RoCE 环境若出现 GPU Direct 报错，追加 `"buffer_device":"cpu"` 到 `kv_connector_extra_config`。

#### 注册节点（DLRouter 在第三台机器 10.201.6.10）

```bash
curl -X POST http://10.201.6.10:8000/nodes/add \
  -H "Content-Type: application/json" \
  -d '{"url": "http://10.201.6.4:8001", "role": "PREFILL"}'

curl -X POST http://10.201.6.10:8000/nodes/add \
  -H "Content-Type: application/json" \
  -d '{"url": "http://10.201.6.5:8001", "role": "DECODE"}'

curl http://10.201.6.10:8000/nodes/status
```

---

### 4.6 跨机部署方案二：LMCacheConnector（无 IB/RoCE）

机器间只有普通以太网时，使用 LMCache（Redis）作为 KV 中转：

```bash
# 先在任意机器上启动 Redis（示例 IP=10.201.6.100）
docker run -d --name redis --network=host redis:latest

# P 节点追加参数（完整 docker run 参考 4.5，替换 --kv-transfer-config 即可）
--kv-transfer-config \
'{"kv_connector":"LMCacheConnector","kv_role":"kv_producer","kv_connector_extra_config":{"url":"redis://10.201.6.100:6379"}}'

# D 节点追加参数
--kv-transfer-config \
'{"kv_connector":"LMCacheConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"url":"redis://10.201.6.100:6379"}}'
```

> Redis 需要足够内存（大模型 KV Cache 数十 GB），建议设置 `maxmemory-policy allkeys-lru` 防止 OOM。注册节点和验证流程同 4.5。

---

## 5. 与 MACA 部署对比

| 项目 | MACA 部署 | NVIDIA 部署 |
|---|---|---|
| 容器设备参数 | `--device=/dev/mxcd` 等 | `--gpus all` |
| Docker 镜像 | `cr.metax-tech.com/.../vllm-metax:...` | `vllm/vllm-openai:latest` |
| PD 连接管理 | LMDeploy `PDConnectionPool`（路由器侧 RDMA）| vLLM `--kv-transfer-config`（实例侧连接）|
| 路由器 PD 配置 | 需要 `--lmdeploy_pd_*` 参数 | 无需额外参数，connector 由 vLLM 实例管理 |
| serving_strategy | `distserve` | `distserve` |

---

## 6. 常见问题

- **PD 节点注册后无流量**：确认节点 `role` 字段，P 节点必须为 `PREFILL`，D 节点为 `DECODE`。
- **Decode 节点重算 KV（未加速）**：检查 P/D 两端的 `--kv-transfer-config` 是否使用同一 connector 和正确的 `kv_rank`。
- **NixlConnector 跨机连接失败**：
  - 检查 IB 设备：`ibv_devices` 应有输出，`ibv_devinfo` 确认端口 Active。
  - 确认 `kv_ip` 填写的是本机对外 IP（不是 127.0.0.1）。
  - 确认 `kv_port`（默认 14579）在两台机器防火墙之间放行。
  - RoCE 环境下若出现 GPU Direct 报错，追加 `"buffer_device":"cpu"` 到 `kv_connector_extra_config`。
- **模型名不匹配**：先请求 `http://<节点>:800x/v1/models`，以返回的 `id` 为准。
- **节点反复掉线**：检查网络连通性和 `DLROUTER_HEALTH_CHECK_TIMEOUT` 环境变量（PD 场景建议设为 60s+）。
