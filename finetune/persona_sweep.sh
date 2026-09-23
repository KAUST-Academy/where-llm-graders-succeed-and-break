#!/bin/bash
# Persona sweep (§5): does the un-tuned base COLLAPSE under a strict-grader
# persona while the bd fine-tune RESISTS it? Generalises the Qwen-7B result
# (FT-A07bd-{base,lora}-{strict,rigorous,exacting}) to the other models.
#
# EVAL ONLY -- the bd adapters already exist. One vLLM server is started per
# model and ALL persona x arm evals run against it (6 evals off a single load,
# instead of paying the model-load cost 6 times).
#
# Usage (GPU node, empty GPU -- check nvidia-smi first):
#   bash finetune/persona_sweep.sh 14b                       # 3 personas x base/lora
#   bash finetune/persona_sweep.sh llama strict              # just one persona
#   bash finetune/persona_sweep.sh 30b strict rigorous exacting
#
#   Gemma is NOT here: vLLM can't apply its adapter -> use eval_gemma_persona.sh
#   (add --batch-size 8 there to cut ~3-4h/eval down to ~30 min).
#
# Neutral is already covered by the main bd runs (FT-<TAG>bd-{base,lora}.xlsx),
# so only strict/rigorous/exacting are swept here.
#
# Re-runnable: any eval whose .xlsx already exists is skipped, so an expired
# allocation just means relaunching the same command.
#
# Knobs: PORT TP CONCURRENCY
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/ibex/user/shaebiyy/conda-environments/gemma4-vllm/bin/python}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
CONCURRENCY="${CONCURRENCY:-32}"
# RECIPE=bd (default) or marks -> selects the split, adapter, prompt-style and
# tag infix. Default bd keeps every existing call unchanged.
RECIPE="${RECIPE:-bd}"
case "$RECIPE" in
  bd)    SPLIT=finetune/data_bd/eval_students.json; STYLE=breakdown; INFIX=bd; ADP_SFX=-bd ;;
  marks) SPLIT=finetune/data/eval_students.json;    STYLE=marks;     INFIX="";  ADP_SFX="" ;;
  *) echo "RECIPE must be bd|marks" >&2; exit 1 ;;
esac
mkdir -p finetune/logs/eval

declare -A MODELS=(
  [7b]="Qwen/Qwen2.5-Coder-7B-Instruct A07"
  [14b]="Qwen/Qwen2.5-Coder-14B-Instruct A14"
  [30b]="Qwen/Qwen3-Coder-30B-A3B-Instruct A30M"
  [llama]="NousResearch/Meta-Llama-3.1-8B-Instruct LL8B"
)

key="${1:?usage: persona_sweep.sh <7b|14b|30b|llama> [personas...]}"
shift || true
PERSONAS=("$@"); [ ${#PERSONAS[@]} -eq 0 ] && PERSONAS=(strict rigorous exacting)
read -r HF_ID TAG <<<"${MODELS[$key]}"
ADAPTER="finetune/adapters/${key}${ADP_SFX}"
[ -f "$ADAPTER/adapter_model.safetensors" ] || { echo "missing adapter: $ADAPTER" >&2; exit 1; }

LOG="finetune/logs/eval/vllm_persona_${key}_$(date +%H%M).log"
echo "#### $key ($HF_ID) persona sweep: [${PERSONAS[*]}] x {base,lora} ####"
echo "== starting vLLM (base + adapter) -> $LOG =="
LORA_NAME=ft nohup bash finetune/serve_lora.sh "$HF_ID" "$ADAPTER" "$PORT" "$TP" \
    > "$LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
  echo "== stopping vLLM (pid $SERVER_PID) =="
  kill "$SERVER_PID" 2>/dev/null || true
  # vllm spawns an EngineCore child that outlives a plain kill
  pkill -9 -f 'EngineCore' 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

evalrun() {  # $1=model(name or 'ft')  $2=tag  $3=persona
  local out="finetune/results/$2.xlsx"
  if [ -f "$out" ]; then echo "== [$(date +%H:%M)] $2 : SKIP (exists) =="; return; fi
  echo "== [$(date +%H:%M)] $2 ($3) : start =="
  $PY finetune/eval_vllm.py --model "$1" --prompt-style "$STYLE" \
      --strictness "$3" --eval-students "$SPLIT" --concurrency "$CONCURRENCY" \
      --api-base "http://localhost:${PORT}/v1" --tag "$2"
  echo "== [$(date +%H:%M)] $2 : exit $? =="
}

for p in "${PERSONAS[@]}"; do
  evalrun "$HF_ID" "FT-${TAG}${INFIX}-base-$p" "$p"
  evalrun ft       "FT-${TAG}${INFIX}-lora-$p" "$p"
done

echo
echo "== DONE ($RECIPE). Results: =="
ls -la finetune/results/FT-${TAG}${INFIX}-*-*.xlsx 2>/dev/null
