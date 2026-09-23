#!/bin/bash
# Launch a vLLM OpenAI-compatible server for one DeepSeek model.
#
# Sibling of serve_qwen.sh / serve_qwen_480B.sh, for the DeepSeek family added
# in the model-family expansion (run IDs L-S* and L-T*). Same CLI surface, so
# any row from computer_vision_results/ablation_runs.xlsx can point at it unchanged.
#
# Args:
#   $1  HuggingFace model id (required)
#   $2  port            (default 8000)
#   $3  tensor parallel (default 4 -- one per A100 in the allocation)
#
# Environment-variable overrides (all optional):
#   MAX_LEN   max model context length         (default 32768)
#   DP        data-parallel size               (default 1)
#   EP        1 to enable expert parallelism   (default 0)
#   MEM_UTIL  gpu memory utilization fraction  (default 0.92)
#   MAX_SEQS  max concurrent sequences         (default 256)
#
# Recipes (Ibex A100 80GB nodes):
#
#   DeepSeek-Coder-V2-Lite-Instruct (16B MoE, 64 experts, BF16, ~31 GB):
#     bash serve_deepseek.sh deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct 8000 4
#     - 16 attention heads, so TP must divide 16: TP=4 gives 4 heads/GPU.
#     - Defaults are fine; the model is small enough that most of the
#       allocation ends up as KV cache.
#
#   DeepSeek-Coder-V2-Instruct-FP8 (236B MoE, 160 experts, ~237 GB) on 8xA100:
#     MEM_UTIL=0.90 MAX_SEQS=64 \
#       bash serve_deepseek.sh neuralmagic/DeepSeek-Coder-V2-Instruct-FP8 8000 8
#     - 128 attention heads and 160 routed experts: both divide by 8 (and by 4).
#     - The checkpoint is per-tensor FP8 with activation_scheme="static". A100
#       is sm80 and has no native FP8 tensor cores, so vLLM falls back to a
#       Marlin weight-only kernel and ignores the activation scales. That is a
#       memory win, not a speed win, and it is the same fallback path the
#       Qwen3-480B-FP8 run (L-R*) used on these nodes.
#     - If that path ever fails to load, the BF16 checkpoint
#       (deepseek-ai/DeepSeek-Coder-V2-Instruct, ~471 GB) still fits on
#       8x80GB=640GB; drop MEM_UTIL to ~0.95 and expect a much longer load.
#
# Notes:
#   - DeepSeek-V2 uses MLA (kv_lora_rank=512), so KV cache per token is far
#     smaller than a comparable dense model -- 32k context is cheap here.
#   - Both models declare max_position_embeddings=163840 via YaRN scaling;
#     MAX_LEN=32768 stays well inside that and matches the Qwen runs.
#   - First launch downloads weights to the HuggingFace cache. Pre-warm with
#     `huggingface-cli download <id>` to keep the wait outside the job.

set -euo pipefail

MODEL="${1:?usage: serve_deepseek.sh <model> [port] [tp]}"
PORT="${2:-8000}"
TP="${3:-4}"

MAX_LEN="${MAX_LEN:-32768}"

# MoE / multi-engine knobs (off by default)
DP="${DP:-1}"
EP="${EP:-0}"

# Memory knobs (override these if you hit OOM at sampler warmup)
MEM_UTIL="${MEM_UTIL:-0.92}"
MAX_SEQS="${MAX_SEQS:-256}"

# Weight dtype. bfloat16 is the only correct choice for the paper's runs: it is
# what these models were trained in, and it matches the A100 Qwen baselines.
# vLLM hard-gates bf16 to compute capability >= 8.0 (platforms/cuda.py:478), so
# on a V100 (sm70) it must be float16 -- which is fine for a plumbing smoke but
# NOT for a result that goes in the paper: fp16 has 5 exponent bits to bf16's 8,
# and this is a bf16-trained MoE, so overflow would be indistinguishable from a
# genuine strict-persona collapse.
DTYPE="${DTYPE:-bfloat16}"

EXTRA_ARGS=(
  --tensor-parallel-size "$TP"
  --max-model-len "$MAX_LEN"
  --max-num-seqs "$MAX_SEQS"
  --dtype "$DTYPE"
  --gpu-memory-utilization "$MEM_UTIL"
  --enable-prefix-caching
  --port "$PORT"
  --host 0.0.0.0
  --served-model-name "$MODEL"
)

# DeepSeek-V2 ships its modelling code in the repo (DeepseekV2ForCausalLM).
EXTRA_ARGS+=(--trust-remote-code)

# Escape hatch for one-off vLLM flags (diagnostics, backend overrides). Word-split
# on purpose so callers can pass several. Example -- forbid the unbounded
# whitespace that xgrammar's JSON grammar otherwise allows between tokens:
#   VLLM_EXTRA_ARGS='--structured-outputs-config {"disable_any_whitespace":true}'
if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS+=(${VLLM_EXTRA_ARGS})
fi

if [[ "$DP" -gt 1 ]]; then
  EXTRA_ARGS+=(--data-parallel-size "$DP")
fi

if [[ "$EP" == "1" ]]; then
  EXTRA_ARGS+=(--enable-expert-parallel)
fi

echo "== Launching vLLM =="
echo "  model:           $MODEL"
echo "  port:            $PORT"
echo "  tensor parallel: $TP"
echo "  data parallel:   $DP"
echo "  expert parallel: $EP"
echo "  dtype:           $DTYPE"
echo "  max model len:   $MAX_LEN"
echo "  max num seqs:    $MAX_SEQS"
echo "  gpu mem util:    $MEM_UTIL"
echo "  endpoint:        http://$(hostname):$PORT/v1"
echo "  total GPUs:      $((TP * DP))"
echo

exec vllm serve "$MODEL" "${EXTRA_ARGS[@]}"
