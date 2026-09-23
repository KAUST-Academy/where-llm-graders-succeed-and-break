#!/bin/bash
# Generic HF-path evaluation runner -- the fallback twin of run_vllm_eval.sh,
# for models whose LoRA adapter vLLM cannot apply (Gemma-4 today). Uses
# eval_lora.py (batched transformers generate()).
#
#   bash finetune/run_hf_eval.sh gemma marks              # base + lora, neutral
#   bash finetune/run_hf_eval.sh gemma bd                 # base + lora, neutral
#   bash finetune/run_hf_eval.sh gemma bd strict rigorous exacting   # persona sweep
#
# Per (variant, persona) it evals BASE then LORA -> finetune/results/
#   FT-<TAG><bd>-{base,lora}[-<persona>].xlsx   (neutral omits the suffix)
# Any eval whose .xlsx already exists is SKIPPED, so reruns are cheap/resumable.
#
# Runs with a clean LD_LIBRARY_PATH (the vLLM CUDA-13 compat libs break HF torch).
# Batch-8 note: batched HF generation shifts absolute MAE by ~0.1 vs batch-1
# (left-padding numerics); base and lora shift together, so comparisons hold.
#
# Knobs: PY BS ADAPTER (override adapter path)
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/ibex/user/shaebiyy/conda-environments/gemma4-vllm/bin/python}"
BS="${BS:-8}"
RUN=(env -u LD_LIBRARY_PATH PYTORCH_ALLOC_CONF=expandable_segments:True "$PY")
GEMMA_SNAP=google/gemma-4-E4B-IT

declare -A MODELS=(
  [7b]="Qwen/Qwen2.5-Coder-7B-Instruct A07"
  [14b]="Qwen/Qwen2.5-Coder-14B-Instruct A14"
  [30b]="Qwen/Qwen3-Coder-30B-A3B-Instruct A30M"
  [llama]="NousResearch/Meta-Llama-3.1-8B-Instruct LL8B"
  [gemma]="$GEMMA_SNAP GE4B"
)

key="${1:?usage: run_hf_eval.sh <7b|14b|30b|llama|gemma> <marks|bd> [personas...]}"
variant="${2:?usage: run_hf_eval.sh <key> <marks|bd> [personas...]}"
shift 2 || true
PERSONAS=("$@"); [ ${#PERSONAS[@]} -eq 0 ] && PERSONAS=(neutral)
read -r HF_ID TAG <<<"${MODELS[$key]}"

case "$variant" in
  marks) STYLE=marks;     INFIX="";   SPLIT=finetune/data/eval_students.json;    TOKENS=256 ;;
  bd)    STYLE=breakdown; INFIX="bd"; SPLIT=finetune/data_bd/eval_students.json; TOKENS=1024 ;;
  *) echo "variant must be marks|bd" >&2; exit 1 ;;
esac

# Adapter: new convention finetune/adapters/<key>[-bd]; fall back to the legacy
# gemma4-e4b-* names from the pre-unification runs. ADAPTER env overrides both.
DEFAULT_ADAPTER="finetune/adapters/${key}${INFIX:+-bd}"
LEGACY_ADAPTER=""
[ "$key" = gemma ] && LEGACY_ADAPTER="finetune/adapters/gemma4-e4b-${INFIX:-marks}"
ADAPTER="${ADAPTER:-}"
if [ -z "$ADAPTER" ]; then
  if   [ -f "$DEFAULT_ADAPTER/adapter_model.safetensors" ]; then ADAPTER="$DEFAULT_ADAPTER"
  elif [ -n "$LEGACY_ADAPTER" ] && [ -f "$LEGACY_ADAPTER/adapter_model.safetensors" ]; then
    ADAPTER="$LEGACY_ADAPTER"; echo "== using legacy adapter: $ADAPTER =="
  else echo "no adapter found ($DEFAULT_ADAPTER${LEGACY_ADAPTER:+ or $LEGACY_ADAPTER})" >&2; exit 1
  fi
fi

evalrun() {  # $1=adapter-or-empty  $2=tag  $3=persona
  local out="finetune/results/$2.xlsx"
  if [ -f "$out" ]; then echo "== [$(date +%H:%M)] $2 : SKIP (exists) =="; return; fi
  echo "== [$(date +%H:%M)] $2 ($3) : start =="
  "${RUN[@]}" finetune/eval_lora.py --model "$HF_ID" ${1:+--adapter "$1"} \
      --prompt-style "$STYLE" --strictness "$3" --eval-students "$SPLIT" \
      --max-new-tokens "$TOKENS" --attn sdpa --no-4bit --batch-size "$BS" --tag "$2"
  echo "== [$(date +%H:%M)] $2 : exit $? =="
}

for p in "${PERSONAS[@]}"; do
  sfx=""; [ "$p" != neutral ] && sfx="-$p"
  evalrun ""         "FT-${TAG}${INFIX}-base${sfx}" "$p"
  evalrun "$ADAPTER" "FT-${TAG}${INFIX}-lora${sfx}" "$p"
done

echo; echo "== DONE =="; ls -la finetune/results/FT-${TAG}${INFIX}-*.xlsx 2>/dev/null
