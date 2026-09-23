#!/bin/bash
#SBATCH --job-name=x-eval
#SBATCH --account=conf-icl-2026.09.25-albarass
#SBATCH --qos=conf-2026h2
#SBATCH --array=0-29%8
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=finetune/logs/slurm/x-eval-%A_%a.out
#SBATCH --error=finetune/logs/slurm/x-eval-%A_%a.out
#
# The 6-cell cross-exam transfer matrix.
#
# One array task = one adapter. Each task starts ONE vLLM server (base + that
# adapter) and grades BOTH exams' held-out sets through it, so the two cells of
# a row cost one model load instead of two:
#
#                        -> eval cv (114 students)   -> eval intro (208)
#     adapter cv         in-domain (reproduces base)  TRANSFER
#     adapter intro      TRANSFER                     in-domain
#     adapter both       in-domain                    in-domain
#
# Task layout (so a subset can be run first):
#     0-9    adapter trained on cv     <- the headline transfer row, no new
#                                         training needed; run this first
#    10-19   adapter trained on intro
#    20-29   adapter trained on both
#
# Results -> finetune/results/X-<TAG>[bd]-<adapter_ds>-on-<eval_ds>.xlsx
# RESUMABLE: a cell whose xlsx already exists is skipped.
#
#   sbatch finetune/cross_eval.sbatch                 # all 30 -> 60 cells
#   sbatch --array=0-9 finetune/cross_eval.sbatch     # transfer row only
#   INCLUDE_BASE=1 sbatch --array=0-9 finetune/cross_eval.sbatch
#       ^ adds the untuned-base cells (10 adapters x 2 exams = 20 base cells);
#         already-present adapter cells are skipped, so this only does the new work
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p finetune/logs/slurm finetune/logs/eval finetune/results

PY="${PY:-/ibex/user/shaebiyy/conda-environments/gemma4-vllm/bin/python}"
CONCURRENCY="${CONCURRENCY:-32}"
PORT="${PORT:-$(( 8000 + ${SLURM_ARRAY_TASK_ID:-0} ))}"

# model-key -> "HF_ID  short-tag". gemma is absent from run_vllm_eval.sh's map;
# it is included here so the matrix covers all five families.
declare -A MODELS=(
  [7b]="Qwen/Qwen2.5-Coder-7B-Instruct A07"
  [14b]="Qwen/Qwen2.5-Coder-14B-Instruct A14"
  [30b]="Qwen/Qwen3-Coder-30B-A3B-Instruct A30M"
  [llama]="NousResearch/Meta-Llama-3.1-8B-Instruct LL8B"
  [gemma]="google/gemma-4-E4B-IT GE4B"
)

# Task index -> (adapter_dataset, model, variant); cv first.
COMBOS=()
for ads in cv intro both; do
  for key in 7b 14b 30b llama gemma; do
    for v in marks bd; do
      COMBOS+=("$ads $key $v")
    done
  done
done

IDX="${SLURM_ARRAY_TASK_ID:-${IDX:-0}}"
[ "$IDX" -ge "${#COMBOS[@]}" ] && { echo "index $IDX out of range" >&2; exit 1; }
read -r ADS KEY VARIANT <<<"${COMBOS[$IDX]}"
read -r HF_ID TAG <<<"${MODELS[$KEY]}"

case "$ADS" in
  cv)    ADS_SFX="" ;;
  intro) ADS_SFX="-intro" ;;
  both)  ADS_SFX="-both" ;;
esac
case "$VARIANT" in
  marks) STYLE="marks";     BD_INFIX="" ;;
  bd)    STYLE="breakdown"; BD_INFIX="bd" ;;
  *) echo "bad variant '$VARIANT'" >&2; exit 1 ;;
esac

ADAPTER="finetune/adapters/${KEY}"
[ "$VARIANT" = bd ] && ADAPTER="${ADAPTER}-bd"
ADAPTER="${ADAPTER}${ADS_SFX}"

# 30B MoE is ~60 GB in bf16; shard it across both GPUs when serving.
TP="${TP:-1}"
[ "$KEY" = "30b" ] && TP=2

ts() { date "+%m-%d %H:%M"; }
echo "[$(ts)] task=$IDX adapter_ds=$ADS model=$KEY variant=$VARIANT tag=$TAG"
echo "[$(ts)] adapter=$ADAPTER  node=$(hostname)  port=$PORT  TP=$TP"

if [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
  echo "[$(ts)] SKIP: adapter not trained yet ($ADAPTER)" >&2
  exit 0
fi

# Work list: "<served-model-name> <result-infix> <exam>" per cell.
#
# INCLUDE_BASE=1 also grades the UNTUNED base model off the same server. The
# base result depends only on (model, style) -- not on what any adapter trained
# on -- so it is attached to the cv row alone (tasks 0-9) and computed once
# instead of three times. It is what makes the transfer numbers interpretable:
# the repo's IA01 intro runs are bd1 WITH reasoning, a different output
# contract, so they cannot baseline the marks adapters at all.
INCLUDE_BASE="${INCLUDE_BASE:-0}"
NEEDED=()
for eds in cv intro; do
  out="finetune/results/X-${TAG}${BD_INFIX}-${ADS}-on-${eds}.xlsx"
  if [ -f "$out" ]; then echo "[$(ts)] SKIP cell (exists): $out"
  else NEEDED+=("ft ${ADS} ${eds}"); fi
  if [ "$INCLUDE_BASE" = "1" ] && [ "$ADS" = "cv" ]; then
    bout="finetune/results/X-${TAG}${BD_INFIX}-base-on-${eds}.xlsx"
    if [ -f "$bout" ]; then echo "[$(ts)] SKIP base cell (exists): $bout"
    else NEEDED+=("$HF_ID base ${eds}"); fi
  fi
done
[ ${#NEEDED[@]} -eq 0 ] && { echo "[$(ts)] all cells present, nothing to do."; exit 0; }

LOG="finetune/logs/eval/xeval_${TAG}${BD_INFIX}_${ADS}_$(date +%m%d_%H%M).log"
echo "[$(ts)] starting vLLM (base + adapter) -> $LOG"
# setsid: own process group so cleanup kills exactly this task's server tree;
# an unscoped `pkill -f EngineCore` would kill sibling array tasks' servers too.
LORA_NAME=ft setsid nohup bash finetune/serve_lora.sh "$HF_ID" "$ADAPTER" "$PORT" "$TP" \
    > "$LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
  echo "[$(ts)] stopping vLLM (pgid $SERVER_PID)"
  kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
  sleep 3
  kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

RC=0
for spec in "${NEEDED[@]}"; do
  read -r SERVED WHICH eds <<<"$spec"
  # The eval split always comes from the exam being GRADED, never from what the
  # adapter was trained on -- that is what keeps every cell of a column measured
  # on identical students.
  split="finetune/data${BD_INFIX:+_bd}"
  [ "$eds" != cv ] && split="${split}_${eds}"
  split="${split}/eval_students.json"

  echo
  echo "[$(ts)] ==== $TAG${BD_INFIX}: $WHICH -> grading $eds ===="
  if $PY finetune/eval_vllm.py --model "$SERVED" --prompt-style "$STYLE" \
        --dataset "$eds" --eval-students "$split" \
        --concurrency "$CONCURRENCY" \
        --api-base "http://localhost:${PORT}/v1" \
        --output-xlsx "finetune/results/X-${TAG}${BD_INFIX}-${WHICH}-on-${eds}.xlsx" \
        --tag "X-${TAG}${BD_INFIX}-${WHICH}-on-${eds}"; then
    echo "[$(ts)] OK   $WHICH -> $eds"
  else
    echo "[$(ts)] FAIL $WHICH -> $eds" >&2; RC=1
  fi
done
exit $RC
