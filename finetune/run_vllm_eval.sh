#!/bin/bash
# Evaluate a fine-tuned adapter the FAST way: start one vLLM server hosting the
# base model + the LoRA adapter, grade the held-out split against both, stop.
#
# Replaces the eval half of run_finetune.sh / run_breakdown_finetune.sh (which
# use HuggingFace generate() at batch size 1). Typical speedup is 10-30x.
#
# Usage:
#   bash finetune/run_vllm_eval.sh <model-key> <variant> [adapter_path]
#     model-key : 7b | 14b | 30b
#     variant   : marks | bd
#     adapter   : optional override (default finetune/adapters/<key>[-bd])
#
# e.g.
#   bash finetune/run_vllm_eval.sh 7b bd
#   bash finetune/run_vllm_eval.sh 7b marks finetune/adapters/7b/checkpoint-188
#
# Knobs: PORT TP CONCURRENCY GPU_UTIL MAX_LEN
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

PY="${PY:-/ibex/user/shaebiyy/conda-environments/gemma4-vllm/bin/python}"
PORT="${PORT:-8000}"
TP="${TP:-1}"                       # tensor-parallel GPUs for serving
CONCURRENCY="${CONCURRENCY:-32}"

declare -A MODELS=(
  [7b]="Qwen/Qwen2.5-Coder-7B-Instruct A07"
  [14b]="Qwen/Qwen2.5-Coder-14B-Instruct A14"
  [30b]="Qwen/Qwen3-Coder-30B-A3B-Instruct A30M"
  [llama]="NousResearch/Meta-Llama-3.1-8B-Instruct LL8B"
)

key="${1:?usage: run_vllm_eval.sh <7b|14b|30b> <marks|bd> [adapter_path]}"
variant="${2:?usage: run_vllm_eval.sh <7b|14b|30b> <marks|bd> [adapter_path]}"
read -r HF_ID TAG <<<"${MODELS[$key]}"

case "$variant" in
  marks) STYLE="marks";     INFIX="";   SPLIT="finetune/data/eval_students.json" ;;
  bd)    STYLE="breakdown"; INFIX="bd"; SPLIT="finetune/data_bd/eval_students.json" ;;
  *) echo "variant must be 'marks' or 'bd'" >&2; exit 1 ;;
esac
ADAPTER="${3:-finetune/adapters/${key}${INFIX:+-bd}}"

mkdir -p finetune/logs/eval
LOG="finetune/logs/eval/vllm_server_${key}${INFIX}_$(date +%H%M).log"
echo "== Starting vLLM (base + adapter) -> $LOG =="
LORA_NAME=ft nohup bash finetune/serve_lora.sh "$HF_ID" "$ADAPTER" "$PORT" "$TP" \
    > "$LOG" 2>&1 &
SERVER_PID=$!
# Always take the server down, even if an eval fails or we're interrupted.
cleanup() {
  echo "== Stopping vLLM (pid $SERVER_PID) =="
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "   (tail -f $LOG to watch it load; eval_vllm.py waits for it)"

echo
echo "== Eval BASE -> finetune/results/FT-${TAG}${INFIX}-base.xlsx =="
$PY finetune/eval_vllm.py --model "$HF_ID" --prompt-style "$STYLE" \
    --eval-students "$SPLIT" --concurrency "$CONCURRENCY" \
    --api-base "http://localhost:${PORT}/v1" --tag "FT-${TAG}${INFIX}-base"

echo
echo "== Eval FINE-TUNED -> finetune/results/FT-${TAG}${INFIX}-lora.xlsx =="
$PY finetune/eval_vllm.py --model ft --prompt-style "$STYLE" \
    --eval-students "$SPLIT" --concurrency "$CONCURRENCY" \
    --api-base "http://localhost:${PORT}/v1" --tag "FT-${TAG}${INFIX}-lora"

echo
echo "Done. Both MAEs printed above."
