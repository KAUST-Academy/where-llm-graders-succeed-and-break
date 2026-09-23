#!/bin/bash
#SBATCH --job-name=ft-cross
#SBATCH --account=conf-icl-2026.09.25-albarass
#SBATCH --qos=conf-2026h2
#SBATCH --array=0-19%10
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=40
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=finetune/logs/slurm/ft-cross-%A_%a.out
#SBATCH --error=finetune/logs/slurm/ft-cross-%A_%a.out
#
# Cross-exam LoRA study: train the intro and combined adapters.
#
# The advance (cv) adapters already exist and are NOT retrained -- an adapter
# trained on cv never saw intro data, so it is already a valid transfer subject.
# That is why this is 20 trainings, not 30:
#
#     adapter        marks  bd      trained here?
#     <key>          [x]    [x]     no  -- reused from the original study
#     <key>-intro    [ ]    [ ]     yes -- 5 models x 2 = 10
#     <key>-both     [ ]    [ ]     yes -- 5 models x 2 = 10
#
# ONE TRAINING PER ARRAY TASK, so tasks backfill independently and run
# concurrently instead of queueing behind each other (%10 caps concurrency at
# 10 tasks = 40 GPUs, well inside the QOS's 250-GPU allowance). The conf-2026h2
# QOS carries priority 5000 vs 0 on normal, which moves the start from ~Sep 9
# to ~Aug 31; its one restriction is <=10 CPUs per GPU, hence 40 for 4 GPUs.
#
# SKIP_BUILD=1 is essential here: the datasets are prebuilt, and without it
# every task would rewrite the same train.jsonl underneath the others.
#
# RESUMABLE: a task whose adapter already exists exits immediately, so
# re-submitting after a timeout or node failure only redoes what is missing.
#
#   sbatch finetune/train_cross_exam.sbatch                    # all 20
#   sbatch --array=0-9 finetune/train_cross_exam.sbatch         # intro only
#   sbatch --array=3 finetune/train_cross_exam.sbatch           # one combo
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p finetune/logs/slurm finetune/logs/train

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Task index -> (dataset, model, variant). Order groups intro (0-9) before
# both (10-19) so --array=0-9 is exactly the intro half.
COMBOS=()
for ds in intro both; do
  for key in 7b 14b 30b llama gemma; do
    for v in marks bd; do
      COMBOS+=("$ds $key $v")
    done
  done
done

IDX="${SLURM_ARRAY_TASK_ID:-${IDX:-0}}"
if [ "$IDX" -ge "${#COMBOS[@]}" ]; then
  echo "index $IDX out of range (0..$(( ${#COMBOS[@]} - 1 )))" >&2; exit 1
fi
read -r DS KEY VARIANT <<<"${COMBOS[$IDX]}"

case "$DS" in
  intro) SFX="-intro" ;;
  both)  SFX="-both"  ;;
  cv)    SFX=""       ;;
  *) echo "bad dataset '$DS'" >&2; exit 1 ;;
esac

ADAPTER="finetune/adapters/${KEY}"
[ "$VARIANT" = bd ] && ADAPTER="${ADAPTER}-bd"
ADAPTER="${ADAPTER}${SFX}"

ts() { date "+%m-%d %H:%M"; }
echo "[$(ts)] task=$IDX  dataset=$DS  model=$KEY  variant=$VARIANT"
echo "[$(ts)] node=$(hostname)  job=${SLURM_ARRAY_JOB_ID:-none}_${SLURM_ARRAY_TASK_ID:-none}"

if [ -f "$ADAPTER/adapter_model.safetensors" ]; then
  echo "[$(ts)] SKIP: $ADAPTER already exists."
  exit 0
fi

NGPU="$(nvidia-smi -L 2>/dev/null | wc -l)"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Same hyperparameters as every prior run -- the cross-exam comparison is only
# meaningful with the training config held fixed.
TRAIN_ENV=(EPOCHS=2 NO_4BIT=1 SAVE_STEPS=25 SKIP_BUILD=1 GPUS="$NGPU")
# The 30B is a 128-expert MoE: ~60 GB of bf16 weights, so it needs FSDP to
# shard them (run_finetune.sh rejects FSDP=1 unless NO_4BIT=1 and GPUS>1).
[ "$KEY" = "30b" ] && [ "$NGPU" -gt 1 ] && TRAIN_ENV+=(FSDP=1)

RUNNER=finetune/run_finetune.sh
[ "$VARIANT" = bd ] && RUNNER=finetune/run_breakdown_finetune.sh

echo "[$(ts)] TRAIN -> $ADAPTER  (GPUS=$NGPU)"
if env "${TRAIN_ENV[@]}" DATASET="$DS" bash "$RUNNER" "$KEY"; then
  echo "[$(ts)] OK   $KEY $VARIANT $DS -> $ADAPTER"
else
  rc=$?
  echo "[$(ts)] FAIL $KEY $VARIANT $DS (exit $rc)" >&2
  exit $rc
fi
