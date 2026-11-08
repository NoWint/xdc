# Pipeline-parallel training and Hugging Face inference

This document describes the P1 pipeline path: a real layer-sharded training run
over the existing outbound WebSocket tunnel, plus layer-sharded HF inference.

## Why pipeline parallelism

DiLoCo already pools consumer devices, but every worker trains a full model copy.
A 135M-parameter Llama needs roughly 2.1 GB just for fp32 weights, gradients and
Adam moments; a 1.7B model needs ~27 GB. Pipeline parallelism assigns each worker
only a contiguous slice of layers, so parameters, gradients, Adam moments and
activations are divided at the same time.

For a stage with `P` local parameters, fp32 training memory is approximately:

```
params  = 4P
grads   = 4P
Adam    = 8P          (exp_avg + exp_avg_sq)
total   = 16P + activations + wire buffers
```

The coordinator adds a 25% default headroom (`stage_memory_headroom`) and a fixed
reserve when it plans slices. Stages never load the full checkpoint.

## Model slices

`xdc/model/llama.py` implements a self-contained Llama/Qwen2-style model with
RMSNorm, RoPE, GQA, SwiGLU and a linear LM head. Hugging Face keys are mapped
explicitly:

```
model.embed_tokens.weight                  -> embed_tokens.weight       (stage 0)
model.layers.{i}.*                         -> layers.{i-local}.*
model.norm.weight                          -> norm.weight               (last stage)
lm_head.weight                             -> lm_head.weight            (last stage)
```

Tied checkpoints omit `lm_head.weight`; the loader derives it from
`model.embed_tokens.weight`. The last stage keeps a synchronized copy, refreshed
from the committed embedding at the start of each round.

Stage files are ordinary safetensors with **local** keys. A worker receives only
its slice, never the full model. The stage file size, local parameter count and
optimizer bytes are reported in job metrics.

## Round protocol

The coordinator drives one round at a time:

1. Select live workers and plan contiguous `[start, end)` ranges that fit each
   worker's reported memory budget. `pipeline_stages` is a preference; after a
   worker loss the planner may reduce the stage count on the remaining nodes.
2. Generate or reuse stage safetensors. Untied stages at the latest version are
   reused from the previous round's uploaded checkpoint; tied heads are
   regenerated from the committed embedding.
3. `pipeline_load` each stage: download, build, load local state, apply optional
   LoRA, move to the chosen device. Nothing is applied yet.
4. `pipeline_begin`: invalidate old KV caches, `zero_grad`, mark the round.
5. Forward: microbatch `m`, stages `0..S-1`. Stage 0 accepts `input_ids`; later
   stages accept a hidden-state wire tensor; the last stage accepts `targets` and
   returns the weighted loss.
6. Backward: reverse microbatch order, reverse stages. The last stage computes
   gradients from the loss; earlier stages receive `grad_hidden` and return the
   gradient at their input. Gradients accumulate locally, scaled by
   `1/micro_batches`.
7. `pipeline_finalize`: per-stage gradient clipping (by the configured norm),
   AdamW step, parameter hash, optimizer safetensors upload.
8. `pipeline_export`: upload the stage's local parameters. The coordinator
   assembles the global checkpoint by replacing the stage's layer range.
   Every stage's `param_hash` is checked against the run before commit.
9. Commit: version += 1, new global checkpoint, optional evaluation, metric row.

There is no commit before all stages finish. If any RPC fails, the coordinator
calls `pipeline_abort`, resets the attempt, and re-plans. A worker that dies
mid-round leaves no applied weights because the step only happens in
`pipeline_finalize`, which cannot run for a silently broken stage.

## Memory and wire accounting

Every round records per stage:

- `param_bytes`, `optimizer_bytes`, `grad_bytes`
- `trainable_parameters`, `parameters`, `device`
- `activation_bytes` and `round_wall_s`

Hidden tensors travel as fp16 by default (`activation_dtype`). For a batch of
`B` sequences, `T` tokens and hidden size `H`, one hop is `2*B*T*H` bytes.
Microbatches are processed sequentially, so the activation buffer is bounded by
one microbatch, not the whole round.

## Fault recovery

- Tunnel disconnect during load/forward/backward/finalize → round error, abort,
  retry with remaining workers.
- Worker loss → `inference_candidates`/planning exclude it; a smaller plan is
  considered.
- Optimizer state is persisted per stage so a re-loaded stage can continue Adam
  rather than restarting it.
- Failed rounds retain `round_attempts`; three attempts fail the job with the
  last error. There is no silent rollback of a partially committed global state
  because commit is atomic (`global.safetensors` is replace-renamed).

## LoRA

`lora_rank > 0` wraps every `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj` and `down_proj` with `LoRALinear`. Base weights (including embeddings,
norms and the LM head) are frozen; only `lora_A`/`lora_B` are trainable and
optimized. Stage files can contain adapter keys; workers load the base slice
first, apply LoRA, then load the adapter keys.

The global checkpoint contains both base and adapter tensors. Inference applies
LoRA before loading, so the same checkpoint is usable without a separate adapter
file. Packed 4-bit QLoRA is not implemented.

## Hugging Face inference

`InferenceEngine.resolve` supports both the built-in TinyGPT and `kind: "hf"`
jobs. For HF:

- `config.json` is parsed into `LlamaConfig` (`head_dim=0` means
  `hidden_size / num_attention_heads`).
- `tokenizer.json` is loaded through the Rust `tokenizers` library. No
  `trust_remote_code` or Python tokenizer module is executed.
- Prompt tokens are tokenized, stages run with KV caches on the owning workers,
  and streamed bytes are produced by incremental `decode()`.
- Stage files are reused from the training job's latest stage checkpoints when
  the inference checkpoint is the latest global checkpoint. This avoids
  rebuilding 100 MB+ slices per request.
- For non-streaming responses `bytes_b64` is the generated UTF-8 text; `tokens`
  is the generated token count.

Tied models need one exception: the final stage's `lm_head` is regenerated from
the global embedding because the last training stage's copy is one optimizer step
behind after commit.

## Limits

- No interleaved 1F1B schedule: microbatches are processed sequentially. This
  keeps the protocol simple and memory bounded, at the cost of pipeline bubbles.
- No tensor parallelism, sequence parallelism or activation recomputation.
- Stage placement is one stage per worker per job, so a worker cannot host two
  stages of the same job.
- The planner uses reported available memory and compute scores. Two processes on
  one physical host share real memory, so the sum of their budgets can exceed the
  host; the 75% fraction and reserve are heuristics, not a hard cgroup.
- Islands (block-boundary traffic reduction across networks) and intra-node
  CPU/GPU hybrid stage placement are not implemented yet.
