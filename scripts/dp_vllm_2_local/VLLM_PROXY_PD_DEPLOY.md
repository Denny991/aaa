# vLLM 官方 Proxy PD 分离部署指南

## 架构

```
客户端
  │
  ▼
disagg_proxy (port 10002, HTTP)
  │  ← P/D 节点通过 ZMQ (port 30002) 自动注册
  │
  ├─── 1. prefill 请求 (max_tokens=1) ──▶ P 节点 (port 8100)
  │                                          │ KV cache via NCCL P2P
  └─── 2. decode 请求 (原始请求) ──────▶ D 节点 (port 8200)
```

proxy 做的事：
1. P/D 节点启动后通过 ZMQ 向 proxy `30002` 端口 ping 注册
2. 客户端请求到 `10002` 端口
3. proxy 构造特殊格式 request_id（含 P/D 的 ZMQ 地址），先发 P 做 prefill（max_tokens=1），再发 D 做 decode
4. 返回 D 节点响应给客户端

---

export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=ALL

## 依赖安装

在运行 proxy 的机器上（使用 PJLab 内网源）：

```bash
python3 -m pip install quart pyzmq msgpack aiohttp --ignore-installed blinker \
    -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ \
    --trusted-host mirrors.i.h.pjlab.org.cn

python3 -m pip install py-spy \
    -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ \
    --trusted-host mirrors.i.h.pjlab.org.cn
```

python3 -m  pip install sglang-router -i http://mirrors.i.h.pjlab.org.cn/repository/pypi-tsinghua/simple --trusted-host mirrors.i.h.pjlab.org.cn

pip install nixl -i http://mirrors.i.h.pjlab.org.cn/repository/pypi-tsinghua/simple --trusted-host mirrors.i.h.pjlab.org.cn
pip install ray -i http://mirrors.i.h.pjlab.org.cn/repository/pypi-tsinghua/simple --trusted-host mirrors.i.h.pjlab.org.cn

python3 -m pip install ray -i http://mirrors.i.h.pjlab.org.cn/repository/pypi-tsinghua/simple --trusted-host mirrors.i.h.pjlab.org.cn
---

## 部署步骤

### 1. 修改 P/D 脚本，加上 proxy 注册配置

在 `--kv-transfer-config` 的 JSON 里增加 `kv_connector_extra_config`，
告诉节点 proxy 的 ZMQ 地址和自己的 HTTP 端口：

**P 节点** `run_p_kimi_pd.sh`，将 `--kv-transfer-config` 改为：
```bash
--kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":1e9,"kv_ip":"10.102.98.166","kv_port":14579,"kv_connector_extra_config":{"proxy_ip":"10.102.98.166","proxy_port":30001,"http_port":8100}}'
```

**D 节点** `run_d_kimi_pd.sh`，将 `--kv-transfer-config` 改为：
```bash
--kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":1e9,"kv_ip":"10.102.97.183","kv_port":14579,"kv_connector_extra_config":{"proxy_ip":"10.102.98.166","proxy_port":30001,"http_port":8200}}'
```

> proxy 跑在 P 机器（`10.102.98.166`）上，脚本已按此配置更新。

### 2. 启动 proxy

```bash
python3 /vllm-workspace/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd.py

# /vllm-workspace/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd.py

python3 /vllm-workspace/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd.py
```

proxy 启动后监听：
- `0.0.0.0:10001` — HTTP，接收客户端请求
- `0.0.0.0:30001` — ZMQ ROUTER，接收 P/D 节点注册 ping

### 3. 启动 P 节点

在 P 机器上：
```bash
bash run_p_kimi_pd.sh
```

P 节点启动后会自动每 5 秒向 proxy `30001` 发 ZMQ ping 注册自己。
proxy 日志出现：
```
🔵Add [HTTP:10.102.98.166:8100, ZMQ:10.102.98.166:14579]
```

### 4. 启动 D 节点

在 D 机器上：
```bash
bash run_d_kimi_pd.sh
```

proxy 日志出现：
```
🔵Add [HTTP:10.102.97.183:8200, ZMQ:10.102.97.183:14579]
```

---

## 发送请求

P/D 注册完成后，请求直接发到 proxy：

```bash
curl http://10.102.98.166:10001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "kimi-k2.5-prefill",
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 100,
        "stream": false
    }'
```

proxy 日志会打印：
```
handle_request count: 0, [HTTP:10.102.98.166:8100, ZMQ:...] 👉 [HTTP:10.102.97.183:8200, ZMQ:...]
```

---

## 注意事项

| 问题 | 说明 |
|------|------|
| P/D 注册超时 | proxy 5 秒内没收到 ping 会移除节点，节点必须持续运行 |
| proxy 崩溃 | P/D 实例不受影响，重启 proxy 后节点自动重新注册 |
| request_id 格式 | proxy 自动构造，不需要手动处理 |
| 直接 curl P/D | ❌ 仍然会报 `request_id does not contain hostname and port` |
| stream | proxy 支持，`"stream": true` 即可 |

---

## 停止服务

```bash
pkill -f "disagg_proxy"
pkill -f "vllm serve.*8100"
pkill -f "vllm serve.*8200"
```

---

**日期**：2026-03-20
**proxy 路径**：`/vllm-workspace/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd.py`


```shell
curl http://100.98.189.44:10001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3-4B",
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 100,
        "stream": false
    }'
curl -s http://100.100.254.204:10001/v1/completions   -H "Content-Type: application/json"   -d '{
    "model": "Qwen3-4B",
    "prompt": "中国的首都是",
    "max_tokens": 50,
    "temperature": 0
  }'
<!-- {"id":"cmpl-___prefill_addr_100.101.93.77:21001___decode_addr_100.101.93.77:22001_7144c4dc9df1458c9e6e9f1608670c0f","object":"text_completion","created":1774868857,"model":"Qwen3-4B","choices":[{"index":0,"text":"北京，对吗？ 是的，中国的首都是北京。北京位于中国北部，是中华人民共和国的首都，也是中国的政治、文化和国际交往中心。北京有许多著名的历史遗迹和现代建筑，如故宫、天安门广场、","logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null,"prompt_logprobs":null,"prompt_token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":3,"total_tokens":53,"completion_tokens":50,"prompt_tokens_details":null},"kv_transfer_params":null}root@test-lt0310-tj5mb-173024-worker-0:~#  -->

# text completions（简单续写）
curl -s http://100.100.254.204:10001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-4B",
    "prompt": "中国的首都是",
    "max_tokens": 50,
    "temperature": 0
  }'

# chat completions（关闭 thinking）
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
    "chat_template_kwargs": {"enable_thinking": false}
  }'

curl -s http://100.100.254.204:10001/v1/chat/completions    -H "Content-Type: application/json"   -d '{
    "model": "Qwen3-4B",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手，请用中文回答。"},
      {"role": "user", "content": "你好，请介绍一下自己"}
    ],
    "max_tokens": 200,
    "temperature": 0.6,
    "chat_template_kwargs": {"enable_thinking": false}
  }'

---

## 同容器双卡 PD 分离部署总结

**日期**：2026-03-30

### 架构概览

```
同一容器内:
  GPU 0 → P 节点 (Prefill, port 8100, ZMQ 21001)
  GPU 1 → D 节点 (Decode,  port 8200, ZMQ 22001)
          ↑ NCCL P2P 走 NVLink/PCIe
  Proxy (port 10001/30001) 运行在外部机器
```

### 遇到的问题及修复

| 问题 | 根因 | 修复 |
|------|------|------|
| D 节点死等 `ncclRecv` 超时 | P/D 两端 `tensor_id` 不匹配（各自生成 `-{uuid8}` 后缀） | 添加 `normalize_tensor_key()` 剥离本地 uuid 后缀 |
| `chatcmpl-` 格式死锁 | 正则只处理 `cmpl-` 格式，未覆盖 `chatcmpl-` | 正则改为 `r'^(?:chat)?cmpl-(.+?)(?:-0)?-[0-9a-f]{8}$'` |
| 输出内容完全错误 | `kv_cache[0]` 破坏张量维度，导致 KV 未正确注入 | 改回 `layer = kv_cache`（不加 `[0]` 索引） |
| NCCL 通信卡死 | `NCCL_IB_DISABLE=0` 在容器内尝试走 IB 失败 | 强制 `NCCL_IB_DISABLE=1` + `NCCL_CUMEM_ENABLE=0` |

### 关键配置

**NCCL 环境变量**（两端必须一致）:
```bash
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_DEBUG=WARN
```

**D 节点必须禁用 prefix caching**:
```bash
--no-enable-prefix-caching
```

### 核心修改文件

| 文件 | 修改点 |
|------|--------|
| `p2p_nccl_connector.py:217` | `layer = kv_cache`（不索引） |
| `p2p_nccl_connector.py:220,307` | 调用 `normalize_tensor_key()` 归一化 tensor_id |
| `p2p_nccl_connector.py:505-522` | 新增 `normalize_tensor_key()` 静态方法 |
| `p2p_nccl_engine.py` | 三阶段调试日志（Q1/Q2/Q3）|
| `run_d_local_2gpu.sh:65` | `--no-enable-prefix-caching` |

### 同步命令

修改后需同步到 vLLM 安装目录:
```bash
cp p2p_nccl_connector.py /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/p2p/
cp p2p_nccl_engine.py /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/p2p/
```

### 验证

```bash
# text completions
curl -s http://100.100.254.204:10001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-4B", "prompt": "中国的首都是", "max_tokens": 50, "temperature": 0}'

# 预期输出: "北京，对吗？是的，中国的首都是北京..."
```