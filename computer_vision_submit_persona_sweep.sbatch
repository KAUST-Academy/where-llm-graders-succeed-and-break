#!/bin/bash
# One SLURM array job = one model's full 5-persona sweep (L-<letter>01..05).
#
# Each array task is self-contained: it gets its own node + GPUs, starts its own
# vLLM server, points a single computer_vision_grade_with_local.py client at it, and tears the
# server down. That mirrors how the Qwen sweeps were run (serve_qwen.sh: "one
# model per job", one client per server) rather than sharing one server between
# personas, so greedy decoding stays free of batch-composition effects.
#
# Persona order matches the existing L-N / L-Q / L-R sweeps exactly:
#   01 neutral   02 strict   03 lenient   04 rigorous   05 exacting
#
# ---------------------------------------------------------------------------
# Usage
#
#   L-S — DeepSeek-Coder-V2-Lite-Instruct (16B MoE, BF16, 4xA100):
#     sbatch --export=ALL,MODEL=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct,\
# MODEL_TAG=deepseek-coder-v2-lite,RID_PREFIX=L-S,TP=4 computer_vision_submit_persona_sweep.sbatch
#
#   L-T — DeepSeek-Coder-V2-Instruct-FP8 (236B MoE, 8xA100). Asking for 8 GPUs
#   is what routes the job to `gpu_wide` (the only A100 partition with 8-GPU
#   nodes); don't name the partition. Slower per call, so more wall-clock:
#     sbatch --gres=gpu:a100:8 --cpus-per-task=64 \
#            --mem=400G --time=2-00:00:00 \
#            --export=ALL,MODEL=neuralmagic/DeepSeek-Coder-V2-Instruct-FP8,\
# MODEL_TAG=deepseek-coder-v2-236b,RID_PREFIX=L-T,TP=8,MEM_UTIL=0.90,MAX_SEQS=64 \
#            computer_vision_submit_persona_sweep.sbatch
#   NOTE: L-T is blocked until the ~237 GB checkpoint has somewhere to live --
#   $HOME has ~76 GB free and there is no scratch allocation yet.
#
#   L-M07 — Qwen2.5-Coder-14B strict on 1xA100, completing the 14B's matched
#   strict/neutral pair (neutral = L-A14). MAX_OUT=4096 matches L-A14 and L-M05
#   (the sweep default is 8192); RID_BASE=5 numbers it after L-M01..06:
#     sbatch --array=1 --gres=gpu:a100:1 --cpus-per-task=10 --mem=128G \
#            --export=ALL,MODEL=Qwen/Qwen2.5-Coder-14B-Instruct,MODEL_TAG=qwen2.5-coder-14b,\
# RID_PREFIX=L-M,RID_BASE=5,TP=1,MAX_OUT=4096 computer_vision_submit_persona_sweep.sbatch
#
# Run a subset of personas with --array (e.g. --array=1 for strict only).
# ---------------------------------------------------------------------------

# Ibex routes to a partition automatically from --gres/--time; naming one
# explicitly is denied ("specifying the 'gpu' partition is not permitted").
# 4xA100 lands in `gpu`, 8xA100 in `gpu_wide`. Do not add --partition here.
#SBATCH --job-name=ag-sweep
#SBATCH --account=pi-albarass
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.out

set -uo pipefail

REPO=/home/shiekhmf/Research/autoGrader
cd "$REPO"

: "${MODEL:?set MODEL=<hf id> via --export}"
: "${MODEL_TAG:?set MODEL_TAG=<short name used in the run tag> via --export}"
: "${RID_PREFIX:?set RID_PREFIX=<e.g. L-S> via --export}"
TP="${TP:-4}"
# MAX_OUT caps the grader's output tokens (default 8192, the cross-family sweeps'
# setting; the Qwen2.5-Coder baselines and L-M05/L-M07 used 4096).
MAX_OUT="${MAX_OUT:-8192}"

PERSONAS=(neutral strict lenient rigorous exacting)
# PERSONAS_OVERRIDE replaces the persona list (space-separated, indexed by array
# task). It contains SPACES, so it must travel via `export` + --export=ALL, never
# inline in --export=...  (the sbatch --export separator traps documented below).
# Used by the L-MP minimal-pair series (presets defined in computer_vision_grade_with_local.py).
if [[ -n "${PERSONAS_OVERRIDE:-}" ]]; then
  read -r -a PERSONAS <<< "$PERSONAS_OVERRIDE"
fi
PERSONA="${PERSONAS[$SLURM_ARRAY_TASK_ID]}"
# RID_BASE offsets the run number so several submissions can share one series
# prefix without colliding (e.g. L-BD: qwen32 RID_BASE=0 -> 01/02, llama70
# RID_BASE=2 -> 03/04, ...). Default 0 keeps the historical numbering.
RID_BASE="${RID_BASE:-0}"
RID=$(printf "%s%02d" "$RID_PREFIX" $((RID_BASE + SLURM_ARRAY_TASK_ID + 1)))

# Default prompt components for this paper: reference solution + guidelines +
# rubric breakdown, no chain-of-thought, temperature 0, single run, all 570.
# BD=0 drops the rubric breakdown (--no-breakdown) -- the L-BD mechanism series:
# the strict persona's "never give partial credit" clause conflicts with the
# breakdown's mandated partial-credit scale, so strict+bd0 tests the paper's
# proposed collapse mechanism (see maintainer notes).
BD="${BD:-1}"
BREAKDOWN_ARG=--with-breakdown
[[ "$BD" == "0" ]] && BREAKDOWN_ARG=--no-breakdown
TAG="${RID}_m-${MODEL_TAG}_sol1_gd1_rs0_bd${BD}_str-${PERSONA}_t00_n1"
OUT="$REPO/computer_vision_results/results/${RID}__${TAG}.xlsx"
TRACKER="$REPO/computer_vision_results/ablation_runs.xlsx"

# STUDENTS=<subset> turns this into a throwaway smoke run (e.g. STUDENTS=1-5).
# It writes outside computer_vision_results/results and outside the tracker, on purpose:
# computer_vision_grade_with_local.py skips students that already have a score, so a smoke run
# that wrote into the real workbook would leave its rows -- including any
# produced by a bug the smoke was meant to catch -- permanently baked into the
# paper's run, silently skipped by the full sweep afterwards.
STUDENTS="${STUDENTS:-}"
if [[ -n "$STUDENTS" ]]; then
  TARGET_ARGS=(--students "$STUDENTS")
  OUT="$REPO/local_logs/smoke_${RID}__${TAG}.xlsx"
  TRACKER="$REPO/local_logs/smoke_tracker.xlsx"
  # computer_vision_grade_with_local.py:1002 otherwise defaults this to computer_vision_results/variance/<tag>.csv
  # -- i.e. a throwaway run drops a file into the real (git-tracked) results tree.
  VARIANCE_ARGS=(--variance-csv "$REPO/local_variance/smoke_${TAG}.csv")
  MODE_DESC="SMOKE students=$STUDENTS (NOT registered)"
else
  TARGET_ARGS=(--all)
  VARIANCE_ARGS=()
  MODE_DESC="full sweep, all 570"
fi

# SAVE_IO=1 dumps every prompt/response under local_prompts/ + local_responses/
# (both gitignored). Diagnostics only -- it writes one file per call, so never
# enable it for a 2,280-call full sweep.
SAVEIO_ARGS=()
[[ -n "${SAVE_IO:-}" ]] && SAVEIO_ARGS=(--save-io)

# NO_WHITESPACE=1 forbids the unbounded whitespace xgrammar's JSON grammar allows
# between tokens (its `[ \n\t]*` is a Kleene star, so a model can pad forever and
# stay grammar-valid -- the suspected cause of DeepSeek truncating at max_tokens).
#
# Build the JSON *here* rather than passing it through --export. Two traps, both
# paid for:
#   1. `sbatch --export` uses ',' as ITS OWN separator, so a JSON value containing
#      a comma is silently truncated at it. Job 48918151 died on
#      '{"backend":"xgrammar"' -- the rest became a bogus second variable.
#   2. disable_any_whitespace is rejected unless backend is explicitly xgrammar or
#      guidance (config/structured_outputs.py:82 in 0.11.2, :64 in 0.19.1). The
#      default backend is "auto", which is NOT accepted -- job 48916283 died on that.
# Hence: no comma crosses --export, and the backend is always pinned.
if [[ -n "${NO_WHITESPACE:-}" ]]; then
  VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-} --structured-outputs-config {\"backend\":\"xgrammar\",\"disable_any_whitespace\":true}"
  export VLLM_EXTRA_ARGS
fi

# Unique port per array task -- and per array JOB. 8000+TASK_ID is not enough:
# every sweep has a task 4, so two concurrent sweeps both want 8004, and ports
# are per-node, not per-job. L-T05 and L-X05 (both task 4) met on gpu102-02:
# L-T's own vLLM was already dead (TP=8 on a 4-GPU alloc), its curl found L-X's
# GLM-4-32B server, printed "vLLM ready after 0s", and spent 7h collecting HTTP
# 404 "model ... does not exist" while grading nothing. Mix in the array job id,
# then walk up to a port nothing is listening on.
port_free() { ! (exec 3<>/dev/tcp/127.0.0.1/"$1") 2>/dev/null; }
PORT=$(( 8000 + (${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}} % 500) * 10 + ${SLURM_ARRAY_TASK_ID:-0} ))
for _ in $(seq 1 50); do
  port_free "$PORT" && break
  echo "-- port $PORT is already in use on $(hostname); trying $((PORT + 1))"
  PORT=$((PORT + 1))
done

mkdir -p slurm_logs local_logs local_variance computer_vision_results/results

# CONDA_ENV selects the serving stack. `conda activate` takes a name or a path.
#   autograder                              -> vLLM 0.11.2 (guided_json live)
#   /ibex/user/shiekhmf/envs/autograder-v19 -> vLLM 0.19.1 (response_format)
# On vLLM versions: 0.11.2 is NOT the safe default, it is the
# deviation -- it is the only stack that actually enforces the JSON grammar, which
# the paper's existing 69 runs almost certainly never had.
# Default flipped 2026-07-19: autograder-v19
# (vLLM 0.19.1, guided_json silently ignored -> unconstrained decoding) is the
# baselines' likely condition and the stack every new-family run used. The old
# default `autograder` (0.11.2) is the ONLY env where the grammar actually
# enforces -- opting INTO it now requires an explicit CONDA_ENV=autograder.
CONDA_ENV="${CONDA_ENV:-/ibex/user/shiekhmf/envs/autograder-v19}"
source /home/shiekhmf/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV" || { echo "!! could not activate conda env: $CONDA_ENV"; exit 1; }

# Weights live on Weka scratch (/ibex/user, 1.5 TB), not $HOME: it keeps the
# 30 GB (L-S) / 237 GB (L-T) checkpoints off the 180 GB home quota, and loading
# off Weka beats NFS badly -- the 30 GB L-S load took ~12 min from home.
# Pre-warm with `HF_HOME=/ibex/user/shiekhmf/hf_cache hf download <id>`. Never
# repoint this at a cache that hasn't been warmed: vLLM re-downloads silently,
# inside the allocation, on the clock.
export HF_HOME="${HF_HOME:-/ibex/user/shiekhmf/hf_cache}"
export VLLM_LOGGING_LEVEL=INFO

# Record the serving stack in every log. Nothing in this repo (no requirements.txt,
# no environment.yml, no tracker column) records which vLLM produced a run, so the
# stack behind the paper's existing 69 results is genuinely unknown -- which is how
# a silently-ignored guided_json went unnoticed for two months. Don't repeat that:
# make every future run state its own provenance.
STACK_VLLM=$(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo "?")
STACK_TORCH=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "?")

echo "=============================================================="
echo " run id      : $RID   (persona: $PERSONA)"
echo " mode        : $MODE_DESC"
echo " model       : $MODEL   (tag: $MODEL_TAG)"
echo " node        : $(hostname)   GPUs: ${CUDA_VISIBLE_DEVICES:-unset}   TP=$TP"
echo " port        : $PORT"
echo " output      : $OUT"
echo " HF_HOME     : $HF_HOME"
echo " conda env   : $CONDA_ENV"
echo " stack       : vllm $STACK_VLLM / torch $STACK_TORCH"
echo "=============================================================="

VLLM_LOG="slurm_logs/vllm-${RID}-${SLURM_JOB_ID}.log"
MEM_UTIL="${MEM_UTIL:-0.92}" MAX_SEQS="${MAX_SEQS:-256}" DTYPE="${DTYPE:-bfloat16}" \
  bash serve_deepseek.sh "$MODEL" "$PORT" "$TP" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# Always take the server down, however this script exits.
cleanup() {
  echo "-- shutting down vLLM (pid $VLLM_PID)"
  kill "$VLLM_PID" 2>/dev/null || true
  # A bare `wait` here is not enough: vLLM can ignore SIGTERM while its TP
  # workers tear down, and the job then holds its GPUs until the wall clock
  # (48896560 sat RUNNING for 13 idle minutes that way). Escalate to SIGKILL,
  # and take the workers with it -- they are children of $VLLM_PID.
  for _ in $(seq 1 15); do
    kill -0 "$VLLM_PID" 2>/dev/null || break
    sleep 1
  done
  pkill -9 -P "$VLLM_PID" 2>/dev/null || true
  kill -9 "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Fail fast if vLLM dies while loading. computer_vision_grade_with_local.py's --wait-for-server
# only polls a socket; it cannot tell "still loading" from "died 90s ago", so a
# crashed server burns the entire allocation. Job 48881830 did exactly that:
# 60 min elapsed, 0 students graded, because all 4 TP workers had already died
# on a CUDA driver mismatch two minutes in. Watch the pid, not just the port.
#
# The pid check is what makes this fast (it caught a crash in 100s), so
# WAIT_SECS only ever backstops a *hang* -- a process that lives but never
# serves. Keep it generous: loading the 30 GB checkpoint off NFS takes ~12 min
# (shards land at 180-250s each) plus ~100s of engine init, and a tight ceiling
# kills a server that was about to work. WAIT_SECS=900 did exactly that to
# 48896560, aborting 5 minutes before the endpoint came up.
#
# Ask the pid BEFORE the port, and make the port prove who it is. The old loop
# curled first and only checked the pid on a *failed* curl, so a curl that
# succeeded on the first try never tested the pid at all -- a dead vLLM plus a
# stranger's server on the same port read as "vLLM ready after 0s". That is how
# L-T05 graded 570 students against L-X05's GLM server (see the PORT block).
# "Something answers /v1/models" is not the question; "is it MY model" is.
WAIT_SECS="${WAIT_SECS:-3600}"
waited=0
while true; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "!! vLLM (pid $VLLM_PID) exited after ${waited}s without serving. Tail of $VLLM_LOG:"
    tail -30 "$VLLM_LOG"
    exit 1
  fi
  resp=$(curl -sf --max-time 5 "http://localhost:${PORT}/v1/models" 2>/dev/null)
  if [[ -n "$resp" ]]; then
    if [[ "$resp" == *"\"${MODEL}\""* ]]; then
      break
    fi
    served=$(printf '%s' "$resp" | grep -oE '"id":"[^"]+"' | head -1 | cut -d'"' -f4)
    echo "!! port ${PORT} on $(hostname) is serving '${served:-<unparseable>}', not '${MODEL}'."
    echo "!! Another job owns this port -- refusing to grade against someone else's model."
    exit 1
  fi
  if [[ "$waited" -ge "$WAIT_SECS" ]]; then
    echo "!! vLLM not ready after ${WAIT_SECS}s; giving up. Tail of $VLLM_LOG:"
    tail -30 "$VLLM_LOG"
    exit 1
  fi
  sleep 10
  waited=$((waited + 10))
done
echo "-- vLLM ready after ${waited}s (serving ${MODEL} on port ${PORT})"

# Server is already up by here, so --wait-for-server is just a safety net.
python computer_vision_grade_with_local.py \
  "${TARGET_ARGS[@]}" \
  "${VARIANCE_ARGS[@]}" \
  "${SAVEIO_ARGS[@]}" \
  --model "$MODEL" \
  --api-base "http://localhost:${PORT}/v1" \
  --wait-for-server 3600 \
  --max-output-tokens "$MAX_OUT" \
  --with-solution \
  --with-guidelines \
  --no-reasoning \
  "$BREAKDOWN_ARG" \
  --strictness "$PERSONA" \
  --temperature 0.0 \
  --runs 1 \
  --tag "$TAG" \
  --output-xlsx "$OUT" \
  --tracker "$TRACKER"
STATUS=$?

echo "-- grader exited with status $STATUS"
exit $STATUS
