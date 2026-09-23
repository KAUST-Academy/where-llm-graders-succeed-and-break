#!/bin/bash
# Launch a vLLM server that hosts a base Qwen model AND a trained LoRA adapter
# at the same time, for fast evaluation of fine-tuning runs.
#
# Because --enable-lora lets one server answer for both, you can grade the
# BASE model and the FINE-TUNED model against the same server without a restart:
#
#   --model <HF_ID>   -> base   (your baseline)
#   --model ft        -> adapter (the fine-tune)
#
# Usage:
#   bash finetune/serve_lora.sh <base_hf_id> <adapter_path> [port] [tp]
# e.g.
#   bash finetune/serve_lora.sh Qwen/Qwen2.5-Coder-7B-Instruct finetune/adapters/7b-bd
#
# Notes:
#   - Prefix caching is the big win here: every prompt shares the same ~18k-char
#     guidelines+rubric+solution prefix, so after the first request that prefill
#     is essentially free.
#   - LORA_RANK must be >= the adapter's r (we train r=16).
#   - MoE caveat: vLLM's LoRA support does not cover MoE expert layers on some
#     architectures. Dense Qwen2.5-Coder 7B/14B are fine; if Qwen3-Coder-30B-A3B
#     is rejected, fall back to finetune/eval_lora.py for that model.
set -euo pipefail

MODEL="${1:?usage: serve_lora.sh <base_hf_id> <adapter_path> [port] [tp]}"
ADAPTER="${2:?usage: serve_lora.sh <base_hf_id> <adapter_path> [port] [tp]}"
PORT="${3:-8000}"
TP="${4:-1}"

LORA_NAME="${LORA_NAME:-ft}"
LORA_RANK="${LORA_RANK:-16}"
MAX_LEN="${MAX_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.92}"

# vLLM binary (override with VLLM=<binary>). This build targets CUDA 13; on an
# older GPU driver the forward-compat libs below are prepended automatically
# (set NO_CUDA_COMPAT=1 to skip, e.g. on a CUDA-13-capable driver).
GV=/ibex/user/shaebiyy/conda-environments/gemma4-vllm
VLLM="${VLLM:-$GV/bin/vllm}"
if [ -z "${NO_CUDA_COMPAT:-}" ]; then
  export LD_LIBRARY_PATH="$GV/cuda-compat:$GV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
fi
# FlashInfer's sampler JITs with nvcc (not installed here); use the native sampler.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

if [ ! -f "$ADAPTER/adapter_config.json" ]; then
  echo "No adapter_config.json in $ADAPTER" >&2
  echo "(If training was interrupted, point at a checkpoint, e.g." >&2
  echo " finetune/adapters/7b/checkpoint-188)" >&2
  exit 1
fi

ARGS=(
  --tensor-parallel-size "$TP"
  --max-model-len "$MAX_LEN"
  --dtype bfloat16
  --gpu-memory-utilization "$GPU_UTIL"
  --enable-prefix-caching
  --enable-lora
  --lora-modules "${LORA_NAME}=${ADAPTER}"
  --max-lora-rank "$LORA_RANK"
  --max-loras 1
  --served-model-name "$MODEL"
  --port "$PORT"
  --host 0.0.0.0
  --trust-remote-code
)

# FORCE_ARCH: override the model's architecture. Needed for multimodal checkpoints
# (Gemma 4 = Gemma4ForConditionalGeneration) whose text weights we fine-tuned:
# forcing the text-only class (Gemma4ForCausalLM) skips the image processor, which
# a text-only checkpoint has no preprocessor_config.json for, and loads no vision
# tower. Example: FORCE_ARCH=Gemma4ForCausalLM
if [ -n "${FORCE_ARCH:-}" ]; then
  ARGS+=(--hf-overrides "{\"architectures\": [\"${FORCE_ARCH}\"]}")
  echo "  FORCE_ARCH : $FORCE_ARCH (text-only override)"
fi

echo "== Launching vLLM (base + LoRA) =="
echo "  base model : $MODEL"
echo "  adapter    : $ADAPTER   (served as '$LORA_NAME')"
echo "  port / tp  : $PORT / $TP"
echo "  max len    : $MAX_LEN"
echo "  endpoint   : http://$(hostname):$PORT/v1"
echo
echo "  eval base : python finetune/eval_vllm.py --model $MODEL --tag <tag>"
echo "  eval lora : python finetune/eval_vllm.py --model $LORA_NAME --tag <tag>"
echo

exec "$VLLM" serve "$MODEL" "${ARGS[@]}"
