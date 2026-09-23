#!/bin/bash
# Marks-only LoRA fine-tuning across three model sizes. TRAINS ONLY --
# evaluate with `bash finetune/run_vllm_eval.sh <key> marks` (vLLM fast path).
#
# Target = TA marks only ({score, bonus}). For the BREAKDOWN-DISTILL variant
# (train on Gemini D01's full grading JSON) use run_breakdown_finetune.sh.
#
# Run this on a GPU node (the same place you run serve_qwen.sh). It processes
# the three models SEQUENTIALLY -- one fits on a single A100 at a time:
#   * Qwen2.5-Coder-7B-Instruct
#   * Qwen2.5-Coder-14B-Instruct
#   * Qwen3-Coder-30B-A3B-Instruct
#
# For each model it: builds the split once (shared) and trains a LoRA adapter.
# Evaluation is a separate step (run_vllm_eval.sh) so the GPUs aren't tied up
# doing batch-size-1 HuggingFace generation.
#
# Usage:
#   bash finetune/run_finetune.sh                 # all three models
#   bash finetune/run_finetune.sh 7b              # just one (7b | 14b | 30b)
#
# Knobs (env vars):
#   SEED=42  TRAIN_FRAC=0.8  EPOCHS=3  MAX_SEQ_LEN=12288
#   Speed knobs (80 GB A100): NO_4BIT=1 (bf16 base, faster matmuls),
#   NO_GRAD_CKPT=1 (skip activation recompute), BATCH=4 GRAD_ACCUM=4
#   (parallel micro-batches; keep BATCH*GRAD_ACCUM=16 for the same effective batch).
#   Fast 14B on 80 GB:  EPOCHS=1 NO_4BIT=1 NO_GRAD_CKPT=1 BATCH=4 GRAD_ACCUM=4
#   Multi-GPU: GPUS=4 (data-parallel). 30B in bf16 additionally needs FSDP=1 --
#   see the note in run_breakdown_finetune.sh -- and a Slurm job with ~256 GB of
#   CPU RAM, because FSDP stages the weights in host memory before sharding them.
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

# Python interpreter (override with PY=<python>).
PY="${PY:-/ibex/user/shaebiyy/conda-environments/gemma4-vllm/bin/python}"
TORCHRUN="${TORCHRUN:-$(dirname "$PY")/torchrun}"
SEED="${SEED:-42}"
TRAIN_FRAC="${TRAIN_FRAC:-0.8}"
EPOCHS="${EPOCHS:-3}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-12288}"
NO_4BIT="${NO_4BIT:-0}"
NO_GRAD_CKPT="${NO_GRAD_CKPT:-0}"
BATCH="${BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
# Checkpoint every N optimizer steps (0 = only at epoch ends). Set e.g.
# SAVE_STEPS=25 so an expired SLURM allocation costs minutes, not the whole
# epoch -- then relaunch with RESUME=1 / --resume to pick up where it stopped.
SAVE_STEPS="${SAVE_STEPS:-0}"
GPUS="${GPUS:-1}"
FSDP="${FSDP:-0}"

# Which linears get a LoRA adapter. Default "all-linear" (what 7B/14B/llama use).
# The 30B MoE is switched to attention-only AUTOMATICALLY in the training loop
# below -- you do not need to pass anything. Set TARGET_MODULES explicitly only
# to override that.
TARGET_MODULES="${TARGET_MODULES:-all-linear}"

# ADAPTER_SUFFIX (e.g. "-lr5e5") writes to finetune/adapters/<key><suffix>
# instead of overwriting the baseline adapter. Mirrors run_breakdown_finetune.sh.
# Use it for controls/sweeps so an existing adapter is never clobbered.
ADAPTER_SUFFIX="${ADAPTER_SUFFIX:-}"

# DATASET selects the exam(s): cv (default, the original study) | intro | both.
# It picks the data dir AND tags the adapter, so cross-exam runs never overwrite
# the cv adapters behind the published FT-* results.
DATASET="${DATASET:-cv}"
case "$DATASET" in
  cv)    DATA_DIR="finetune/data";           DS_SUFFIX="" ;;
  intro) DATA_DIR="finetune/data_intro";     DS_SUFFIX="-intro" ;;
  both)  DATA_DIR="finetune/data_both";      DS_SUFFIX="-both" ;;
  *) echo "DATASET must be cv|intro|both (got '$DATASET')" >&2; exit 1 ;;
esac

# Keep the GLOBAL batch (BATCH*GRAD_ACCUM*GPUS) constant as GPUs scale, so the
# optimizer-step count -- and the learning dynamics -- match the 1-GPU run.
PER_GPU_ACCUM="$GRAD_ACCUM"
if [ "$GPUS" -gt 1 ]; then
  PER_GPU_ACCUM=$(( GRAD_ACCUM / GPUS ))
  [ "$PER_GPU_ACCUM" -lt 1 ] && PER_GPU_ACCUM=1
  LAUNCH=("$TORCHRUN" --standalone --nproc_per_node="$GPUS")
  echo "== Multi-GPU: $GPUS GPUs, per-GPU grad-accum=$PER_GPU_ACCUM "\
"(global batch = $BATCH x $PER_GPU_ACCUM x $GPUS = $(( BATCH * PER_GPU_ACCUM * GPUS ))) =="
else
  LAUNCH=("$PY")
fi

# Assemble optional train flags from the speed knobs.
# NOTE: --target-modules is NOT set here -- it is per-model (see the loop below),
# because the 30B MoE needs a different value and this runner can train several
# models in one invocation.
TRAIN_FLAGS=(--batch-size "$BATCH" --grad-accum "$PER_GPU_ACCUM")
[[ "$NO_4BIT" == "1" ]] && TRAIN_FLAGS+=(--no-4bit)
[[ "$NO_GRAD_CKPT" == "1" ]] && TRAIN_FLAGS+=(--no-gradient-checkpointing)
[[ "$SAVE_STEPS" != "0" ]] && TRAIN_FLAGS+=(--save-steps "$SAVE_STEPS")
# RESUME=1 -> continue from the newest checkpoint-N in the adapter dir (train_lora
# starts fresh when none exist, so this is safe to set chain-wide).
[[ -n "${RESUME:-}" ]] && TRAIN_FLAGS+=(--resume)
if [[ "$FSDP" == "1" ]]; then
  [[ "$NO_4BIT" == "1" ]] || { echo "FSDP=1 needs NO_4BIT=1 (FSDP shards bf16, not 4-bit)." >&2; exit 1; }
  [ "$GPUS" -gt 1 ] || { echo "FSDP=1 needs GPUS>1 (nothing to shard across on one GPU)." >&2; exit 1; }
  TRAIN_FLAGS+=(--fsdp)
fi

# model-key -> "HF_ID  short-tag"
declare -A MODELS=(
  [7b]="Qwen/Qwen2.5-Coder-7B-Instruct A07"
  [14b]="Qwen/Qwen2.5-Coder-14B-Instruct A14"
  [30b]="Qwen/Qwen3-Coder-30B-A3B-Instruct A30M"
  [llama]="NousResearch/Meta-Llama-3.1-8B-Instruct LL8B"
  [gemma]="google/gemma-4-E4B-IT GE4B"
)

WHICH=("${1:-all}")
if [[ "${WHICH[0]}" == "all" ]]; then
  WHICH=(7b 14b 30b llama gemma)
fi

# SKIP_BUILD=1 -> reuse the data already on disk. Required when many trainings
# run concurrently (e.g. a job array): each would otherwise rewrite the SAME
# train.jsonl underneath the others mid-read.
if [ "${SKIP_BUILD:-0}" = "1" ]; then
  echo "== Step 0: SKIPPED (SKIP_BUILD=1), reusing existing data =="
else
  echo "== Step 0: build dynamic split ($DATASET, seed=$SEED frac=$TRAIN_FRAC) =="
  $PY finetune/build_finetune_data.py --dataset "$DATASET" \
      --seed "$SEED" --train-frac "$TRAIN_FRAC"
fi

for key in "${WHICH[@]}"; do
  read -r HF_ID TAG <<<"${MODELS[$key]}"
  ADAPTER="finetune/adapters/${key}${DS_SUFFIX}${ADAPTER_SUFFIX}"
  echo
  echo "########################################################"
  echo "# $key  ($HF_ID)  tag=$TAG"
  echo "########################################################"

  # Per-model LoRA targets. The 30B is a 128-expert MoE: "all-linear" would adapt
  # 48 layers x 128 experts x 3 projections (~18k modules, ~830M trainable params
  # -> ~74 GB) and OOM an 80 GB card hours into the run. Attention-only (~13M
  # params) is the conventional MoE choice. Applied AUTOMATICALLY so a bare
  # `run_finetune.sh` (which defaults to "all" = 7b 14b 30b) doesn't die on the
  # last model. Override by exporting TARGET_MODULES explicitly.
  TM="$TARGET_MODULES"
  if [ "$key" = "30b" ] && [ "$TM" = "all-linear" ]; then
    TM="q_proj,k_proj,v_proj,o_proj"
    echo "== 30B MoE: target-modules -> $TM (all-linear would OOM) =="
  fi

  echo "== Train LoRA ($key) =="
  TRAINLOG="finetune/logs/train/${key}${DS_SUFFIX}_marks_$(date +%m%d_%H%M).log"
  mkdir -p finetune/logs/train
  "${LAUNCH[@]}" finetune/train_lora.py \
      --model "$HF_ID" \
      --data "$DATA_DIR/train.jsonl" \
      --output-dir "$ADAPTER" \
      --seed "$SEED" --epochs "$EPOCHS" --max-seq-len "$MAX_SEQ_LEN" \
      --target-modules "$TM" \
      "${TRAIN_FLAGS[@]}" 2>&1 | tee "$TRAINLOG"

    # Verify the tokenizer aligned the completion mask (fail otherwise).
  MM=$(grep -c "Mismatch between tokenized" "$TRAINLOG" || true)
  if [ "${MM:-0}" -gt 0 ]; then
    echo "FATAL: $MM tokenizer/mask mismatches in $TRAINLOG." >&2
    echo "       -> discard $ADAPTER and re-check the training setup." >&2
    exit 1
  fi
  echo "== Trained ($key) -> $ADAPTER  (mask check ok) =="
done

echo
echo "Training done. This runner TRAINS ONLY. Evaluate with the vLLM path"
echo "(10-30x faster than HuggingFace generate(); grades base + adapter off one"
echo "server):"
echo
for key in "${WHICH[@]}"; do
  echo "    bash finetune/run_vllm_eval.sh $key marks"
done
echo
echo "-> finetune/results/FT-<tag>-{base,lora}.xlsx"
