# XDC 测试教程

这份教程面向第一次接手 XDC 的开发者，覆盖四层验证：

1. 快速单元测试（不启动进程）
2. 真实多进程 e2e（coordinator + worker）
3. Hugging Face / LoRA / pipeline 训练
4. 分层推理、混合设备、island 调度

所有命令都在仓库根目录执行；示例使用 macOS/Linux 的 `bash`，Windows 请把
`source .venv/bin/activate` 换成 `.venv\Scripts\Activate.ps1`。

> 这些测试证明的是**协议、正确性和故障行为**。除特别说明外，多个 worker 都在
> 同一台物理机器上，不能据此声称多机加速。验收边界见
> [acceptance.md](acceptance.md)。

---

## 0. 环境准备

要求：

- Python >= 3.10
- 可选：C++ 编译器（构建 native 聚合扩展）
- 测试 HF 模型需要 `tokenizers` / `transformers` / `huggingface_hub`
- e2e 会启动真实进程，建议至少 4 GB 可用内存和 2 GB 临时磁盘

```bash
python3 -m venv .venv
source .venv/bin/activate

# 开发依赖 + Hugging Face 可选依赖
python -m pip install -e ".[dev,hf]"

# 可选：构建 C++ 扩展。失败不影响测试，会走 Python fallback
python setup.py build_ext --inplace

# 自检：报告 CPU / MPS / CUDA / ROCm / XPU 的真实可用性
xdc doctor --json
```

确认 native 扩展：

```bash
python -c "from xdc import native; print(native.AVAILABLE, native.VERSION)"
```

### 常见安装问题

| 现象 | 处理 |
|---|---|
| `tokenizers` 或 `transformers` 导入失败 | `pip install -e ".[dev,hf]"` |
| native 构建失败 | 检查编译器；不构建也能跑全部 Python 测试 |
| macOS Intel 上 NumPy 版本冲突 | `pyproject.toml` 已按平台固定 NumPy 1.x，不要全局覆盖 |
| Windows 安装 pybind11 失败 | 安装 MSVC Build Tools 后重试 |

---

## 1. 测试分层与耗时

```bash
# 快速：不启动集群进程（模型、crypto、数据、native 等）
make unit
# 等价于
pytest -q -m "not e2e"

# 真实多进程：每个测试启动独立 coordinator + 2/4 个 worker
make e2e
# 等价于
pytest -q -m e2e

# 全部
make test
```

当前基线：**30 passed**，全量约 5–11 分钟，取决于磁盘和 MPS 是否参与。
`make unit` 约 1–2 分钟；pipeline 的集群测试都带 `e2e` 标记，不会拖慢 unit。

常用选择器：

```bash
# 只跑 pipeline 文件里的单元正确性
pytest tests/test_pipeline.py -q -m "not e2e"

# 只跑 pipeline 集群测试
pytest tests/test_pipeline.py -q -m e2e

# 单测一个用例，-s 保留日志
pytest tests/test_pipeline.py::test_pipeline_step_matches_monolithic -q -s

# 只跑不需要 MPS 的 e2e（MPS 用例会自动 skip）
pytest tests/test_pipeline.py -q -m e2e -k "not hybrid"
```

### 测试矩阵

| 文件 / 用例 | 验证内容 |
|---|---|
| `tests/test_crypto.py` | Ed25519 身份、签名、replay、TLS 材料 |
| `tests/test_model.py` | 单进程 / 两 stage 前向、KV cache、checkpoint、FP16、native/Python 聚合 |
| `tests/test_tunnel.py` | tunnel 替换后旧 RPC 不污染新连接 |
| `tests/test_e2e.py` | 两 worker DiLoCo、checkpoint 恢复、worker 被杀、TLS、鉴权、分片下载 |
| `test_llama_logits_and_gradients_match_transformers` | 自研 Llama 与 HF `LlamaForCausalLM` logits/gradients 一致 |
| `test_stage_files_really_shard_the_model` | stage 文件小于全量，拼合 logits 一致 |
| `test_pipeline_step_matches_monolithic` | 两 worker 一步 pipeline == 单进程 AdamW（fp32 误差 < 2e-5） |
| `test_pipeline_tied_head_is_synchronized` | tied embedding 的 head 同步 |
| `test_pipeline_lora_freezes_base_weights` | base 权重逐位不变、LoRA 可训练、LoRA 可推理 |
| `test_hf_pipeline_inference_matches_monolithic` | 两 stage + KV cache 的 greedy 文本与单进程完全一致 |
| `test_pipeline_recovers_after_stage_worker_loss` | 第二阶段 worker 中途死亡后降级重规划并完成 |
| `test_hybrid_mps_cpu_pipeline` | stage 0 MPS + stage 1 CPU 训练与推理 |
| `test_island_placement_constraint_and_inference` | `training.island` 钉住训练与推理 |
| `test_island_blocks_minimize_cross_island_hops` | 4 worker / 2 island，`island_hops <= 1` |

没有对应硬件的测试会自动 skip：

- MPS 用例要求 `torch.backends.mps.is_available()`
- CUDA / ROCm / XPU 验证状态以 `xdc doctor` 和 acceptance 报告为准
- 没有 `transformers` 时，HF 对照测试 skip（自研模型测试仍运行）

---

## 2. 手动最小集群

先初始化一个 demo 集群：

```bash
xdc init --dir ./xdc-demo --host 127.0.0.1 --port 8787
```

会生成：

```text
xdc-demo/
  coordinator.yaml
  worker1.yaml
  worker2.yaml
  data/
  tls/            # 如果加了 --tls
```

开三个终端：

```bash
# 终端 1
xdc coordinator --config ./xdc-demo/coordinator.yaml --access-log

# 终端 2
xdc worker --config ./xdc-demo/worker1.yaml

# 终端 3
xdc worker --config ./xdc-demo/worker2.yaml
```

检查节点和健康状态：

```bash
xdc health --config ./xdc-demo/coordinator.yaml
xdc nodes --config ./xdc-demo/coordinator.yaml
```

两个节点都应显示 `online=true`、`tunnel_connected=true`。没有 tunnel 时，
`xdc nodes` 可能已经显示 HTTP 注册成功，但推理仍会等待真实 WebSocket。

### 跑一次 TinyGPT

```bash
xdc train --config ./xdc-demo/coordinator.yaml \
  --dataset datasets/tiny_stories.txt \
  --rounds 3 --local-steps 10 --batch-size 4 --seq-len 64 \
  --min-workers 2 --max-workers 2 --wait
```

期望：

- `state=completed`
- `version=3`
- `updates_received=2`、`updates_used=2`
- loss 下降（不要求每轮单调）

推理：

```bash
xdc infer --config ./xdc-demo/coordinator.yaml \
  --job-id <job-id> \
  --prompt "Mira found a brass key near" \
  --max-tokens 16 --temperature 0 --no-stream
```

输出会带 `distributed=True stages=2`。

---

## 3. Hugging Face + Pipeline 测试

### 3.1 上传模型

推荐直接从 Hugging Face 拉取 safetensors：

```bash
xdc model pull HuggingFaceTB/SmolLM2-135M-Instruct \
  --name smollm2-135m \
  --config ./xdc-demo/coordinator.yaml
```

只下载 `config.json`、`tokenizer*.json`、`generation_config.json` 和
`safetensors`，不会下载 Python 模型代码，也不会执行 `trust_remote_code`。

离线环境：

```bash
xdc model add /path/to/SmolLM2-135M-Instruct \
  --name smollm2-135m \
  --config ./xdc-demo/coordinator.yaml
```

确认：

```bash
xdc model list --config ./xdc-demo/coordinator.yaml
xdc model show smollm2-135m --config ./xdc-demo/coordinator.yaml
```

### 3.2 调整上传限制

小模型默认 512 MB 足够。大模型 shard 可能超过限制，修改
`xdc-demo/coordinator.yaml`：

```yaml
coordinator:
  max_upload_mb: 2048
```

重启 coordinator 后生效。

### 3.3 LoRA pipeline 训练

```bash
xdc train --config ./xdc-demo/coordinator.yaml \
  --dataset datasets/tiny_stories.txt \
  --hf-model smollm2-135m \
  --parallelism pipeline \
  --rounds 2 \
  --batch-size 1 --seq-len 128 --micro-batches 2 \
  --pipeline-stages 2 \
  --lora-rank 4 --lora-alpha 8 \
  --min-workers 2 --max-workers 2 \
  --eval-every 1 --wait
```

关注输出：

- `stages 2: 0-17@worker-1 17-30@worker-2`（层数会随内存预算变化）
- `state=completed`
- `loss initial -> final` 有下降
- 每个 stage 的 `param_bytes` / `optimizer_bytes`

再看 job 详情：

```bash
xdc job <job-id> --config ./xdc-demo/coordinator.yaml
```

重点字段：

```text
metrics[-1].stage_metrics[*].device
metrics[-1].stage_metrics[*].param_bytes
metrics[-1].stage_metrics[*].optimizer_bytes
metrics[-1].island_hops
stage_plan[*].layer_start / layer_end
```

### 3.4 Pipeline 推理

```bash
xdc infer --config ./xdc-demo/coordinator.yaml \
  --job-id <job-id> \
  --prompt "Once upon a time" \
  --max-tokens 8 --temperature 0 --no-stream
```

第一次请求可能较慢：

- coordinator 要加载 global checkpoint
- 如果训练 stage 文件不可复用，要重新生成 stage safetensors
- MPS 首次 kernel 有 warmup

之后再请求会走 `ResolvedModel` 缓存，明显更快。

### 3.5 混合设备

让 worker1 用 Metal、worker2 用 CPU：

```yaml
# xdc-demo/worker1.yaml
worker:
  backend: mps

# xdc-demo/worker2.yaml
worker:
  backend: cpu
```

重启两个 worker，再跑一次 pipeline job。`stage_metrics[*].device` 应该同时出现
`mps` 和 `cpu`。没有 MPS 的机器不要这样做，测试会自动 skip。

CUDA 机器可把 `backend` 改为 `cuda`；但本仓库尚未在真实 CUDA 硬件上验收，
不要把它写成已验证。

### 3.6 Island 标签

```yaml
# worker1.yaml
worker:
  labels:
    island: rack-a

# worker2.yaml
worker:
  labels:
    island: rack-b
```

两种用法：

```bash
# 1) 钉死：只使用 rack-a 的节点
xdc train ... --island rack-a

# 2) 不钉死：planner 按 island 分成连续 block
xdc train ...
```

job 详情里的 `island_hops` 应 <= island 数量 - 1。推理同样遵守 island 约束和
block 顺序。

---

## 4. 单独跑 benchmark

TinyGPT / DiLoCo：

```bash
python benchmarks/benchmark.py \
  --workdir /tmp/xdc-bench \
  --output benchmarks/latest.json \
  --rounds 3 --local-steps 10 --workers 2
```

SmolLM2 pipeline：

```bash
python benchmarks/smollm2_pipeline.py \
  --model-dir /path/to/SmolLM2-135M-Instruct \
  --rounds 2 --lora-rank 4 --backends cpu,cpu
```

混合设备：

```bash
python benchmarks/smollm2_pipeline.py \
  --model-dir /path/to/SmolLM2-135M-Instruct \
  --rounds 1 --lora-rank 4 --backends mps,cpu
```

脚本会真实启动 coordinator + 2 worker，上传模型和数据集，训练、推理并打印
stage 指标。结果只代表当前主机，不是多机 scaling 结论。

---

## 5. 故障注入

### 5.1 杀 worker

已有自动化用例：

```bash
pytest tests/test_pipeline.py::test_pipeline_recovers_after_stage_worker_loss -q -s
```

手动：

```bash
# 训练启动后
pkill -f "xdc worker --config ./xdc-demo/worker2.yaml"
```

期望：

- 当前 round abort
- `round_attempts` 增加
- 下一轮在剩余节点上降为 1 stage
- job 最终 completed，`version` 继续推进
- 事件里能看到 `pipeline round failed` 和 `pipeline round opened`

### 5.2 杀 coordinator

```bash
pkill -f "xdc coordinator --config ./xdc-demo/coordinator.yaml"
xdc coordinator --config ./xdc-demo/coordinator.yaml
```

已提交的 round 保留；进行中的 round 丢弃。resume 后继续训练即可。

### 5.3 磁盘不足

e2e 和 SMOL 模型会写较多 stage/optimizer 文件。清理：

```bash
rm -rf /tmp/xdc-bench* /tmp/xdc-smollm-* /tmp/xdc-demo
find "${TMPDIR:-/tmp}" -maxdepth 1 -name 'xdc-*' -type d -exec rm -rf {} +
```

检查：

```bash
df -h .
```

---

## 6. 排错表

| 报错 / 现象 | 原因 | 处理 |
|---|---|---|
| `413 upload too large` | shard 超过 `max_upload_mb` | 提高 coordinator 配置并重启 |
| `no feasible pipeline plan` | stage 内存预算不足 | 降低 `seq_len` / `batch_size` / `micro_batches`，或提高 `pipeline_memory_fraction` |
| `only 1 stage fit; min_workers=2` | 节点内存不够切两段 | 降低 `min_workers` 或减小模型/序列 |
| `no inference workers with active tunnels` | worker 还没连上 WS | 等几秒，`xdc nodes` 确认 `tunnel_connected` |
| 推理第一次很慢 | 生成 stage 文件 / MPS warmup | 第二次请求对比；检查磁盘占用 |
| `MPS backend out of memory` | Metal 显存不足 | worker `backend: cpu` |
| hybrid 测试 skip | 当前机器无 MPS | 正常；不需要伪造通过 |
| worker 日志为空 | 进程还没 flush / 启动失败 | 看 pytest temp 下的 `logs/worker-N.log`，或手动前台启动 |
| 端口冲突 | 旧 coordinator 未退出 | `pkill -f 'xdc coordinator --config'` 后重启 |
| `tokenizer.json not found` | 模型目录不完整 | 用 `xdc model pull`，不要只拷 `model.safetensors` |

### 收集 e2e 日志

`tests/conftest.py::Cluster` 会把日志写到测试临时目录：

```text
<tmp>/logs/coordinator.log
<tmp>/logs/worker-1.log
<tmp>/logs/worker-2.log
```

测试失败时 `wait_job` / `wait_nodes` 会把合并日志直接抛到 pytest 输出。也可以在
测试里调用 `cluster.dump_logs()`。

---

## 7. 提交前检查

```bash
python -m ruff check xdc tests benchmarks
python -m compileall -q xdc
make unit
make e2e          # 有硬件/时间时
git status --short
```

确认没有把以下内容提交：

- 真实 token / 私钥 / TLS 证书
- 用户数据或私有 checkpoint
- `*.safetensors` 模型权重
- 测试临时日志
- `/tmp` 绝对路径（除文档示例外）

验收标准：新增能力必须有对应的测试或真实跑测证据；没有硬件的路径保持
`unverified`，不要为了好看改成 `verified`。
