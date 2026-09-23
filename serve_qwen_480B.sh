#!/bin/bash
# Launch a vLLM OpenAI-compatible server for one Qwen model.
#
# Intended workflow on Ibex:
#   1. sbatch a job that starts a Jupyter server on a 4xA100 node (your
#      existing 4a100168h script does this).
#   2. Open the Jupyter URL in your browser.
#   3. In the Jupyter UI, open TWO terminals on that node:
#        Terminal A:  bash serve_qwen.sh Qwen/Qwen2.5-Coder-32B-Instruct
#        Terminal B:  python computer_vision_grade_with_local.py ...   (any row from
#                     computer_vision_results/ablation_runs.xlsx -- the default
#                     --api-base http://localhost:8000/v1 already points at
#                     the same node, so no edits needed.)
#   4. Repeat across your 4 SLURM jobs to run 4 ablations in parallel
#      (one model per job).
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
# Recipes for specific models on 8xA100 80GB:
#
#   Qwen3-Coder-480B-A35B-Instruct-FP8 (MoE, block-FP8 quantized):
#     MEM_UTIL=0.88 MAX_SEQS=64 DP=2 EP=1 \
#       bash serve_qwen.sh Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 8000 4
#     - MUST use TP=4 DP=2 EP=1: TP=8 would give a 320-wide MoE shard
#       that isn't divisible by the block_n=128 FP8 quantization block.
#     - Lower MEM_UTIL + MAX_SEQS to leave room for the sampler warmup;
#       the model fills ~59 GiB/GPU just for weights.
#
#   Qwen2.5-Coder-32B-Instruct or similar dense models on 4xA100:
#     bash serve_qwen.sh Qwen/Qwen2.5-Coder-32B-Instruct
#     (all defaults are fine.)
#
# Notes:
#   - Terminal A stays occupied by vLLM (logs scroll there). To background it
#     instead so the same terminal can run other commands:
#         nohup bash serve_qwen.sh <model> > vllm.out 2>&1 &
#     Then `tail -f vllm.out` to watch it boot.
#   - First launch downloads the model weights (~30-480 GB depending on size)
#     to your HuggingFace cache. Pre-warm with `huggingface-cli download <id>`
#     if you want to avoid that wait time inside the job.

set -euo pipefail

MODEL="${1:?usage: serve_qwen.sh <model> [port] [tp]}"
PORT="${2:-8000}"
TP="${3:-4}"

# Defensive: pick a context length that fits in 4x A100 80GB for any of these
# models. Qwen3-Coder-Next *natively* supports 256k but KV cache for full ctx
# is huge; 32k is a sane default for grading (typical prompt is ~25k chars).
MAX_LEN="${MAX_LEN:-32768}"

# MoE / multi-engine knobs (off by default; harmless for dense models)
DP="${DP:-1}"
EP="${EP:-0}"

# Memory knobs (override these if you hit OOM at sampler warmup)
MEM_UTIL="${MEM_UTIL:-0.92}"
MAX_SEQS="${MAX_SEQS:-256}"

# Speed knobs. bf16 weights, eager attention OFF (use flash). Enable prefix
# caching so repeated prompt prefixes (guidelines, rubric) are cheap.
EXTRA_ARGS=(
  --tensor-parallel-size "$TP"
  --max-model-len "$MAX_LEN"
  --max-num-seqs "$MAX_SEQS"
  --dtype bfloat16
  --gpu-memory-utilization "$MEM_UTIL"
  --enable-prefix-caching
  --port "$PORT"
  --host 0.0.0.0
  --served-model-name "$MODEL"
)

# Some Qwen3 / Qwen3-Next models need trust-remote-code while their classes
# are still in transformers-staging. Harmless for older Qwen2.5 too.
EXTRA_ARGS+=(--trust-remote-code)

# Data parallel (combines with TP; total GPUs used = TP * DP)
if [[ "$DP" -gt 1 ]]; then
  EXTRA_ARGS+=(--data-parallel-size "$DP")
fi

# Expert parallel for MoE models
if [[ "$EP" == "1" ]]; then
  EXTRA_ARGS+=(--enable-expert-parallel)
fi

echo "== Launching vLLM =="
echo "  model:           $MODEL"
echo "  port:            $PORT"
echo "  tensor parallel: $TP"
echo "  data parallel:   $DP"
echo "  expert parallel: $EP"
echo "  max model len:   $MAX_LEN"
echo "  max num seqs:    $MAX_SEQS"
echo "  gpu mem util:    $MEM_UTIL"
echo "  endpoint:        http://$(hostname):$PORT/v1"
echo "  total GPUs:      $((TP * DP))"
echo

# vllm serve <model> [vllm-flags ...]
exec vllm serve "$MODEL" "${EXTRA_ARGS[@]}"