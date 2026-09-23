#!/bin/bash
#SBATCH --job-name=x-persona
#SBATCH --account=conf-icl-2026.09.25-albarass
#SBATCH --qos=conf-2026h2
#SBATCH --array=0-9%6
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=finetune/logs/slurm/x-persona-%A_%a.out
#SBATCH --error=finetune/logs/slurm/x-persona-%A_%a.out
#
# Persona sweep for the POOLED (both-exams) adapters -- the study's main
# adapters now that the intro dataset is in. Reruns the original E2 question on
# them, on BOTH exams: does the untuned base collapse under a strict-grader
# persona while the fine-tune resists?
#
# One array task = one (model, contract). Personas only change the PROMPT, so a
# single vLLM load serves every cell of that row:
#     3 personas x 2 exams x {base, both-adapter} = 12 evals per server load
# (the original persona_sweep.sh paid one model load per persona).
#
# The NEUTRAL reference cells already exist from the transfer matrix
# (X-<TAG>[bd]-{base,both}-on-<exam>.xlsx); this adds the persona cells:
#     X-<TAG>[bd]-{base,both}-on-<exam>-<persona>.xlsx
#
# RESUMABLE: cells whose xlsx exists are skipped.
#
#   sbatch finetune/persona_cross_eval.sbatch          # all 10 rows -> 120 cells
#   sbatch --array=0,1 finetune/persona_cross_eval.sbatch   # just 7b marks+bd
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p finetune/logs/slurm finetune/logs/eval finetune/results

PY="${PY:-/ibex/user/shaebiyy/conda-environments/gemma4-vllm/bin/python}"
CONCURRENCY="${CONCURRENCY:-32}"
PORT="${PORT:-$(( 8100 + ${SLURM_ARRAY_TASK_ID:-0} ))}"
PERSONAS="${PERSONAS:-strict rigorous exacting}"

declare -A MODELS=(
  [7b]="Qwen/Qwen2.5-Coder-7B-Instruct A07"
  [14b]="Qwen/Qwen2.5-Coder-14B-Instruct A14"
  [30b]="Qwen/Qwen3-Coder-30B-A3B-Instruct A30M"
  [llama]="NousResearch/Meta-Llama-3.1-8B-Instruct LL8B"
  [gemma]="google/gemma-4-E4B-IT GE4B"
)

COMBOS=()
for key in 7b 14b 30b llama gemma; do
  for v in marks bd; do COMBOS+=("$key $v"); done
done
IDX="${SLURM_ARRAY_TASK_ID:-${IDX:-0}}"
read -r KEY VARIANT <<<"${COMBOS[$IDX]}"
read -r HF_ID TAG <<<"${MODELS[$KEY]}"

case "$VARIANT" in
  marks) STYLE="marks";     BD_INFIX="" ;;
  bd)    STYLE="breakdown"; BD_INFIX="bd" ;;
esac
ADAPTER="finetune/adapters/${KEY}$([ "$VARIANT" = bd ] && echo -bd)-both"
# ADAPTER_OVERRIDE points at a different adapter directory (e.g. the text-only
# Gemma copies from finetune/make_text_only_adapter.py); result names are unchanged.
ADAPTER="${ADAPTER_OVERRIDE:-$ADAPTER}"
TP=1; [ "$KEY" = "30b" ] && TP=2

ts() { date "+%m-%d %H:%M"; }
echo "[$(ts)] task=$IDX model=$KEY variant=$VARIANT adapter=$ADAPTER node=$(hostname)"

[ -f "$ADAPTER/adapter_model.safetensors" ] || { echo "adapter missing: $ADAPTER" >&2; exit 1; }

# Work list first, so a fully-done row exits before paying the model load.
NEEDED=()   # "<served-model> <arm> <exam> <persona>"
for persona in $PERSONAS; do
  for eds in cv intro; do
    for arm in base both; do
      out="finetune/results/X-${TAG}${BD_INFIX}-${arm}-on-${eds}-${persona}.xlsx"
      if [ -f "$out" ]; then echo "[$(ts)] SKIP (exists): $(basename $out)"
      else
        served="$HF_ID"; [ "$arm" = both ] && served="ft"
        NEEDED+=("$served $arm $eds $persona")
      fi
    done
  done
done
[ ${#NEEDED[@]} -eq 0 ] && { echo "[$(ts)] all persona cells present."; exit 0; }
echo "[$(ts)] ${#NEEDED[@]} cells to run"

LOG="finetune/logs/eval/xpersona_${TAG}${BD_INFIX}_$(date +%m%d_%H%M).log"
# setsid puts the server and every vLLM worker it spawns (EngineCore included)
# in their own process group, so cleanup can kill exactly that group. A bare
# `pkill -f EngineCore` would also hit SIBLING array tasks co-scheduled on the
# same node and kill their servers mid-eval.
LORA_NAME=ft setsid nohup bash finetune/serve_lora.sh "$HF_ID" "$ADAPTER" "$PORT" "$TP" > "$LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
  sleep 3
  kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

RC=0
for spec in "${NEEDED[@]}"; do
  read -r SERVED ARM EDS PERSONA <<<"$spec"
  split="finetune/data${BD_INFIX:+_bd}"
  [ "$EDS" != cv ] && split="${split}_${EDS}"
  split="${split}/eval_students.json"
  out="X-${TAG}${BD_INFIX}-${ARM}-on-${EDS}-${PERSONA}"
  echo; echo "[$(ts)] ==== $out ===="
  if $PY finetune/eval_vllm.py --model "$SERVED" --prompt-style "$STYLE" \
        --dataset "$EDS" --eval-students "$split" --strictness "$PERSONA" \
        --concurrency "$CONCURRENCY" --api-base "http://localhost:${PORT}/v1" \
        --output-xlsx "finetune/results/${out}.xlsx" --tag "$out"; then
    echo "[$(ts)] OK   $out"
  else
    echo "[$(ts)] FAIL $out" >&2; RC=1
  fi
done
exit $RC
