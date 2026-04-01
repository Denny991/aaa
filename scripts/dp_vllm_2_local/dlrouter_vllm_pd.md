# DLRouter vLLM 部署指南

本文档介绍如何使用 DLRouter 代理 vLLM 推理服务，支持两种模式：
- **Hybrid 模式**：标准 vLLM 代理
- **DistServe 模式**：vLLM PD 分离部署

---

## 环境准备

```bash
# 激活 conda 环境
conda activate dlrouter

# 确保依赖已安装
pip install -e .
```

---

## 模式一：Hybrid 模式（标准代理）

适用于标准 vLLM 实例，手动注册节点。

### 1. 启动 vLLM 服务

```bash
# 在 100.103.140.93 上执行
bash /home/liutong/myNode/codes/DLRouter/scripts/run_vllm_standalone.sh
```

> vLLM 将运行在 8000 端口

### 2. 启动 DLRouter

```bash
python3 -m dlrouter \
    --backend vllm \
    --serving_strategy hybrid \
    --server_port 8000 \
    --routing_strategy min_expected_latency
    --disable_cache_status
```

### 3. 注册节点

```bash
# 添加 HYBRID 节点
curl -X POST http://localhost:8000/nodes/add \
    -H "Content-Type: application/json" \
    -d '{"url": "http://100.103.140.93:8000"}'
```

### 4. 测试请求

```bash
curl -s http://100.100.254.204:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3-4B",
        "messages": [
            {"role": "system", "content": "你是一个有帮助的助手，请用中文回答。"},
            {"role": "user", "content": "你好，请介绍一下自己"}
        ],
        "max_tokens": 200,
        "temperature": 0.6,
        "tool_choice": "none"
    }'
```

---

## 模式二：DistServe 模式（PD 分离）

适用于 vLLM PD 分离部署，P/D 节点通过 ZMQ 自动注册。

### 1. 启动 DLRouter

```bash
python3 -m dlrouter \
    --backend vllm \
    --serving_strategy distserve \
    --zmq_discovery_enabled \
    --zmq_discovery_port 30001 \
    --server_port 10001
    --disable_cache_status
```

### 2. 启动 vLLM P 节点

```bash
vllm serve /mnt/shared-storage-user/ailab-sys/caikun/models/Qwen/Qwen3-4B \
    --served-model-name Qwen3-4B \
    --host 0.0.0.0 \
    --port 8100 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    --distributed-executor-backend mp \
    --kv-transfer-config '{
        "kv_connector": "P2pNcclConnector",
        "kv_role": "kv_producer",
        "kv_buffer_size": 2e9,
        "kv_port": 21001,
        "kv_connector_extra_config": {
            "proxy_ip": "100.100.254.204",
            "proxy_port": "30001",
            "http_port": 8100
        }
    }'
```

### 3. 启动 vLLM D 节点

```bash
vllm serve /mnt/shared-storage-user/ailab-sys/caikun/models/Qwen/Qwen3-4B \
    --served-model-name Qwen3-4B \
    --host 0.0.0.0 \
    --port 8200 \
    --tensor-parallel-size 1 \
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
```

> P/D 节点会自动向 DLRouter 的 ZMQ 端口（30001）发送心跳注册。

### 4. 测试请求

```bash
curl -s http://100.100.254.204:10001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3-4B",
        "messages": [
            {"role": "system", "content": "你是一个有帮助的助手，请用中文回答。"},
            {"role": "user", "content": "你好，请介绍一下自己"}
        ],
        "max_tokens": 200,
        "temperature": 0.6,
        "tool_choice": "none",
        "chat_template_kwargs": {"enable_thinking": false}
    }'
```

---

## 常用操作

### 查看节点状态

```bash
curl http://localhost:8000/nodes/status
```

### 移除节点

```bash
curl -X POST http://localhost:8000/nodes/remove \
    -H "Content-Type: application/json" \
    -d '{"url": "http://100.103.140.93:8000"}'
```

---

## 端口说明

| 服务 | 端口 | 说明 |
|------|------|------|
| DLRouter Hybrid | 8000 | 标准代理模式 |
| DLRouter DistServe | 10001 | PD 分离模式 |
| vLLM Hybrid | 8000 | 标准 vLLM 实例 |
| vLLM P 节点 | 8100 | PD 分离 Prefill 节点 |
| vLLM D 节点 | 8200 | PD 分离 Decode 节点 |
| ZMQ 服务发现 | 30001 | P/D 节点注册端口 |

---

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--backend` | 后端类型：`vllm` 或 `lmdeploy` | `lmdeploy` |
| `--serving_strategy` | 服务模式：`hybrid` 或 `distserve` | `hybrid` |
| `--routing_strategy` | 路由策略 | `min_expected_latency` |
| `--zmq_discovery_enabled` | 启用 ZMQ 自动发现 | `False` |
| `--zmq_discovery_port` | ZMQ 监听端口 | `30001` |
| `--server_port` | DLRouter 服务端口 | `8000` |
| `--api_keys` | API 密钥（逗号分隔） | 无 |

---

## 注意事项

1. **Qwen3 模型请求**：需要添加 `"tool_choice": "none"`，否则 vLLM 会报错
2. **Hybrid 与 DistServe 可同时运行**：使用不同端口互不干扰
3. **节点角色**：
   - `role=1`：HYBRID（标准节点）
   - `role=2`：PREFILL（PD 分离预填充节点）
   - `role=3`：DECODE（PD 分离解码节点）
