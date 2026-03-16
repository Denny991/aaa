# MACA + vLLM + DLRouter 部署文档

本文整理了在 MACA 环境下部署 `vLLM` 节点并接入 `DLRouter` 的最小可用流程。

## 1. 前置条件

- 宿主机已安装 Docker，且具备对应设备节点权限（如 `/dev/mxcd`、`/dev/infiniband`）。
- 模型目录已就绪（示例使用 `/datapool/tangzhiyi/data/Qwen3-32B`）。
- `DLRouter` 代码可运行（建议先在项目根目录执行 `pip install -e .`）。

## 2. 镜像说明

- MACA 基础镜像：
  `crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/maca:latest`
- vLLM-MACA 镜像：
  `cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.14.0-maca.ai3.5.3.102-torch2.8-py312-ubuntu22.04-amd64`

## 3. 启动容器

### 3.1 启动 DLRouter 运行容器（可选）

如果你希望在容器内启动 DLRouter，可使用：

```bash
docker run -itd \
  --privileged \
  --ipc host \
  --cap-add SYS_PTRACE \
  --device=/dev/mem \
  --device=/dev/dri \
  --device=/dev/mxcd \
  --device=/dev/infiniband \
  --group-add video \
  --network=host \
  --shm-size 400g \
  --ulimit memlock=-1 \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  -h "$(hostname)" \
  --name lt_maca_proxy \
  -v /mnt:/mnt \
  -v /datapool:/datapool \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -w /datapool/lt \
  --entrypoint /bin/bash \
  "crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/maca:latest"
```

### 3.2 启动 vLLM 容器

```bash
docker run -itd \
  --privileged \
  --ipc host \
  --cap-add SYS_PTRACE \
  --device=/dev/mem \
  --device=/dev/dri \
  --device=/dev/mxcd \
  --device=/dev/infiniband \
  --group-add video \
  --network=host \
  --shm-size 400g \
  --ulimit memlock=-1 \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  -h "$(hostname)" \
  --name lt_maca_vllm \
  -v /mnt:/mnt \
  -v /datapool:/datapool \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -w /datapool/lt \
  --entrypoint /bin/bash \
  "cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.14.0-maca.ai3.5.3.102-torch2.8-py312-ubuntu22.04-amd64"
```

进入容器：

```bash
docker exec -it lt_maca_vllm bash
```

## 4. 启动 vLLM 服务

在 `lt_maca_vllm` 容器内执行：

```bash
vllm serve /datapool/tangzhiyi/data/Qwen3-32B \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16
```

本地自测（在 vLLM 所在机器执行）：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/datapool/tangzhiyi/data/Qwen3-32B",
    "messages": [{"role": "user", "content": "你好，请介绍一下你自己"}]
  }'
```

停止 vLLM：

```bash
pkill -f "vllm serve"
```

## 5. 启动 DLRouter（vLLM 后端）

示例：在 `10.201.6.10` 启动路由服务。

```bash
dlrouter \
  --backend vllm \
  --server_name 10.201.6.10 \
  --server_port 8000 \
  --routing_strategy min_observed_latency \
  --serving_strategy hybrid \
  --log_level DEBUG
```

说明：

- `--backend vllm`：使用 vLLM 适配器。
- `--serving_strategy hybrid`：vLLM 场景下使用该模式即可。

## 6. 注册 vLLM 节点并验证

在另一个终端执行（将 `10.201.6.5` 替换为 vLLM 节点地址）：

```bash
curl -X POST http://10.201.6.10:8000/nodes/add \
  -H "Content-Type: application/json" \
  -d '{"url": "http://10.201.6.5:8000"}'
```

查看节点状态：

```bash
curl http://10.201.6.10:8000/nodes/status
```

通过 DLRouter 转发请求：

```bash
curl http://10.201.6.10:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/datapool/tangzhiyi/data/Qwen3-32B",
    "messages": [{"role": "user", "content": "你好，请介绍一下你自己"}],
    "tool_choice": "none"
  }'
```

## 7. LMDeploy 后端（Hybrid 单节点）对比部署

用于与 vLLM 做性能对比，采用 Hybrid 模式（单节点，无 PD 拆分）。

### 7.1 启动 LMDeploy 容器

```bash
docker run -itd \
   --privileged \
   --ipc host \
   --cap-add SYS_PTRACE \
   --device=/dev/mem \
   --device=/dev/dri \
   --device=/dev/mxcd \
   --device=/dev/infiniband \
   --group-add video \
   --network=host \
   --shm-size 400g \
   --ulimit memlock=-1 \
   --security-opt seccomp=unconfined \
   --security-opt apparmor=unconfined \
   -h "$(hostname)" \
   --name lt_maca \
   -v /mnt:/mnt \
   -v /datapool:/datapool \
   -v /var/run/docker.sock:/var/run/docker.sock \
   -w /datapool/lt \
   --entrypoint /bin/bash \
   crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/maca:latest
```

进入容器：

```bash
docker exec -it lt_maca_lmdeploy bash
```

### 7.2 启动 LMDeploy api_server（Hybrid 模式）

无需 `--role` ，即为标准 Hybrid 单节点：

```bash
lmdeploy serve api_server \
  /datapool/tangzhiyi/data/Qwen3-32B \
  --model-name /datapool/tangzhiyi/data/Qwen3-32B \
  --server-name 10.201.6.5 \
  --server-port 23333 \
  --proxy-url http://10.201.6.10:8000 \
  --backend pytorch \
  --device maca \
  --cache-block-seq-len 16 \
  --tp 4
```


### 7.3 启动 DLRouter（LMDeploy 后端）

```bash
dlrouter \
  --backend lmdeploy \
  --server_name 10.201.6.10 \
  --server_port 8000 \
  --routing_strategy min_observed_latency \
  --serving_strategy hybrid \
  --log_level DEBUG
```

### 7.4 注册节点并验证

```bash
# 自动注册节点
# curl -X POST http://10.201.6.10:8000/nodes/add \
#   -H "Content-Type: application/json" \
#   -d '{"url": "http://10.201.6.5:23333"}'

# 查看节点状态
curl http://10.201.6.10:8000/nodes/status
```

### 7.5 性能对比 benchmark

```bash
# 仅仅 `backend  vllm lmdeploy` 不同
MODEL_NAME="/datapool/tangzhiyi/data/Qwen3-32B"
TOKENIZER_PATH=/datapool/tangzhiyi/data/Qwen3-32B
SEED=5
NUM_PROMPTS=500
INPUT_LENGTH=8196
PREFIX_LENGTH=6553
OUTPUT_LENGTH=2048
for i in `seq 1 1`
do
    echo "doing"
    python /datapool/tangzhiyi/hetero_ppu/dev/lmdeploy/benchmark/profile_restful_api.py \
    --host 10.201.6.10 \
    --port 8000  \
    --backend lmdeploy \
    --dataset-name random \
    --dataset-path /datapool/tangzhiyi/data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --random-input-len ${INPUT_LENGTH} \
    --random-output-len ${OUTPUT_LENGTH} \
    --model ${MODEL_NAME} \
    --random-range-ratio 0.5 \
    --tokenizer ${TOKENIZER_PATH} \
    --seed ${i} \
    --num-prompts ${NUM_PROMPTS}
done
```

## 8. 常见问题

- 节点注册成功但不可用：
  先检查 `curl http://<vllm节点>:8000/health` 是否返回 200。
  如：curl http://10.201.6.10:8000/nodes/status
- `nodes/status` 无节点或反复掉线：
  检查 DLRouter 与 vLLM 之间网络连通性、端口、防火墙策略。
- 模型名不匹配：
  先请求 `http://<vllm节点>:8000/v1/models`，以返回的 `id` 为准填写 `model` 字段。
