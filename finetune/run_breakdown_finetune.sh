#!/bin/bash
# Breakdown-distill LoRA fine-tuning: train a model to emit
# {score, bonus, task_breakdown} -- the per-task awards distilled from Gemini
# D01. No `reasoning` field (build_breakdown_data.py drops it by default; pass
# --with-reasoning there if you ever want the prose back).
#
# Pipeline:
#   0. build_breakdown_data.py  -> finetune/data_bd/{train,eval}.jsonl
#      (reuses finetune/data/split_meta.json so it's the SAME held-out students
#       as the marks-only run; needs a prior build_finetune_data.py run.)
#   1. train_lora.py            -> finetune/adapters/<key>-bd
# TRAINS ONLY. Evaluate with: bash finetune/run_vllm_eval.sh <key> bd
#
# Usage (on an 80 GB A100, GPU empty -- check nvidia-smi first):
#   bash finetune/run_breakdown_finetune.sh 14b
#
# Knobs: SEED EPOCHS MAX_SEQ_LEN NO_4BIT NO_GRAD_CKPT BATCH GRAD_ACCUM FSDP
#   Recommended 14B/80 GB:  EPOCHS=2 NO_4BIT=1 BATCH=1 GRAD_ACCUM=16
#   Recommended 30B/4x80 GB: GPUS=4 FSDP=1 NO_4BIT=1 EPOCHS=2
#     The 30B needs FSDP=1. Under plain DDP every rank holds a full copy, and
#     61 GB of bf16 weights + ~10 GB of LoRA/optimiser state does not leave room
#     for 16k-token activations on an 80 GB card. FSDP shards the weights across
#     the GPUs (~15 GB each) and keeps bf16, so the 30B stays directly comparable
#     to the bf16 7B/14B runs instead of being the one 4-bit outlier.
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

# Python interpreter (override with PY=<python>).
PY="${PY:-/ibex/user/shaebiyy/conda-environments/gemma4-vllm/bin/python}"
TORCHRUN="${TORCHRUN:-$(dirname "$PY")/torchrun}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-2}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"   # breakdown completions are long; keep headroom
NO_4BIT="${NO_4BIT:-1}"
NO_GRAD_CKPT="${NO_GRAD_CKPT:-0}"
BATCH="${BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
# Checkpoint every N optimizer steps (0 = only at epoch ends). Set e.g.
# SAVE_STEPS=25 so an expired SLURM allocation costs minutes, not the whole
# epoch -- then relaunch with RESUME=1 / --resume to pick up where it stopped.
SAVE_STEPS="${SAVE_STEPS:-0}"
GPUS="${GPUS:-1}"                      # GPUs for TRAINING (eval stays on 1)
FSDP="${FSDP:-0}"                      # 1 = shard the model across GPUS (needed for 30B bf16)

# Which linears get a LoRA adapter. Default "all-linear" (what 7B/14B used).
# For the 30B MoE, pass TARGET_MODULES="q_proj,k_proj,v_proj,o_proj": "all-linear"
# would adapt 48 layers x 128 experts x 3 projections = ~18k modules (~830M
# trainable params -> ~74 GB, OOM on one 80 GB card, and 18k tiny matmuls per
# step makes it crawl). Attention-only is ~13M params, ~65 GB, and is the
# conventional choice for MoE.
TARGET_MODULES="${TARGET_MODULES:-all-linear}"

# Applied AUTOMATICALLY for the 30B MoE (rather than left as a comment the caller
# has to remember): "all-linear" there adapts ~18k expert projections (~830M
# params) and OOMs an 80 GB card hours in. Export TARGET_MODULES to override.
if [ "${1:-14b}" = "30b" ] && [ "$TARGET_MODULES" = "all-linear" ]; then
  TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"
  echo "== 30B MoE: target-modules -> $TARGET_MODULES (all-linear would OOM) =="
fi

# LR / rank knobs for hyperparameter sweeps. The 2e-4 default is a 7B-scale
# value; larger models generally want a lower LR (they start from a stronger
# base, so the same schedule over-corrects -- visible as a growing positive
# score bias). ADAPTER_SUFFIX keeps sweep runs from overwriting each other.
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-$((LORA_R * 2))}"   # keep alpha = 2r unless overridden
ADAPTER_SUFFIX="${ADAPTER_SUFFIX:-}"

# DATASET selects the exam(s): cv (default) | intro | both. See run_finetune.sh.
DATASET="${DATASET:-cv}"
case "$DATASET" in
  cv)    BD_DIR="finetune/data_bd";       DS_SUFFIX="" ;;
  intro) BD_DIR="finetune/data_bd_intro"; DS_SUFFIX="-intro" ;;
  both)  BD_DIR="finetune/data_bd_both";  DS_SUFFIX="-both" ;;
  *) echo "DATASET must be cv|intro|both (got '$DATASET')" >&2; exit 1 ;;
esac
# MoE + 4-bit: PEFT would upcast the (unquantized) expert weights to fp32 and OOM.
SKIP_KBIT_UPCAST="${SKIP_KBIT_UPCAST:-0}"

# Keep the GLOBAL batch (BATCH*GRAD_ACCUM*GPUS) constant as GPUs scale, so the
# number of optimizer steps -- and thus the learning dynamics -- match the
# single-GPU run. We divide grad-accum by GPUS rather than growing the batch.
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

TRAIN_FLAGS=(--batch-size "$BATCH" --grad-accum "$PER_GPU_ACCUM"
             --target-modules "$TARGET_MODULES"
             --lr "$LR" --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA")
[[ "$NO_4BIT" == "1" ]] && TRAIN_FLAGS+=(--no-4bit)
[[ "$SKIP_KBIT_UPCAST" == "1" ]] && TRAIN_FLAGS+=(--skip-kbit-upcast)
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

declare -A MODELS=(
  [7b]="Qwen/Qwen2.5-Coder-7B-Instruct A07"
  [14b]="Qwen/Qwen2.5-Coder-14B-Instruct A14"
  [30b]="Qwen/Qwen3-Coder-30B-A3B-Instruct A30M"
  [llama]="NousResearch/Meta-Llama-3.1-8B-Instruct LL8B"
  [gemma]="google/gemma-4-E4B-IT GE4B"
)
key="${1:-14b}"
read -r HF_ID TAG <<<"${MODELS[$key]}"
# ADAPTER_SUFFIX (e.g. "-lr5e5") keeps sweep runs in their own directory instead
# of overwriting the baseline adapter.
ADAPTER="finetune/adapters/${key}-bd${DS_SUFFIX}${ADAPTER_SUFFIX}"

# SKIP_BUILD=1 -> reuse the data already on disk. Required when many trainings
# run concurrently (e.g. a job array): each would otherwise rewrite the SAME
# train.jsonl underneath the others mid-read.
if [ "${SKIP_BUILD:-0}" = "1" ]; then
  echo "== Step 0: SKIPPED (SKIP_BUILD=1), reusing existing data =="
else
  echo "== Step 0: build breakdown-distill data ($DATASET, from the bd1 teacher run) =="
  $PY finetune/build_breakdown_data.py --dataset "$DATASET"
fi

echo
echo "########################################################"
echo "# $key  ($HF_ID)  tag=${TAG}bd   BREAKDOWN-DISTILL"
echo "########################################################"

echo "== Train LoRA ($key, breakdown) =="
TRAINLOG="finetune/logs/train/${key}${DS_SUFFIX}_bd_$(date +%m%d_%H%M).log"
mkdir -p finetune/logs/train
"${LAUNCH[@]}" finetune/train_lora.py \
    --model "$HF_ID" \
    --data "$BD_DIR/train.jsonl" \
    --output-dir "$ADAPTER" \
    --seed "$SEED" --epochs "$EPOCHS" --max-seq-len "$MAX_SEQ_LEN" \
    "${TRAIN_FLAGS[@]}" 2>&1 | tee "$TRAINLOG"

# Verify the tokenizer aligned the completion mask (fail otherwise).
MM=$(grep -c "Mismatch between tokenized" "$TRAINLOG" || true)
if [ "${MM:-0}" -gt 0 ]; then
  echo "FATAL: $MM tokenizer/mask mismatches in $TRAINLOG." >&2
  echo "       -> discard $ADAPTER and re-check the training setup." >&2
  exit 1
fi

echo
echo "== Training done -> $ADAPTER  (mask check ok) =="
echo
echo "This runner TRAINS ONLY. Evaluate with the vLLM path (10-30x faster than"
echo "HuggingFace generate(); it grades base + adapter off one server):"
echo
echo "    bash finetune/run_vllm_eval.sh $key bd"
echo
echo "-> finetune/results/FT-${TAG}bd-{base,lora}.xlsx"
