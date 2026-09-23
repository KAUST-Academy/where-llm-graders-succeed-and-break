#!/bin/bash
# One SLURM array job = one model's persona sweep on the Stage 2 (Intro to AI)
# exam. Structured to match computer_vision_submit_persona_sweep.sbatch (the Stage 3 sweep):
# same env-var interface, same RID/TAG/output conventions, same smoke-run
# isolation, same tracker back-fill. Differences are dataset-driven and marked
# STAGE-2 below.
#
# Each array task is self-contained: its own node + GPU, its own vLLM server,
# one introduction_to_ai_grade_with_local.py client, then teardown. Persona order matches
# the Stage 3 sweep exactly:
#   01 neutral   02 strict   03 lenient   04 rigorous   05 exacting
#
# ---------------------------------------------------------------------------
# Usage
#
#   Full neutral+strict pair for one model (the published configuration):
#     sbatch --array=0,1 --export=ALL,MODEL=Qwen/Qwen2.5-Coder-32B-Instruct,\
# MODEL_TAG=qwen25-coder-32b,RID_PREFIX=IA introduction_to_ai_submit_persona_sweep.sbatch
#
#   Smoke first (writes OUTSIDE the real results tree and skips the tracker):
#     sbatch --array=1 --export=ALL,STUDENTS=1-5,MODEL=...,MODEL_TAG=... \
#            introduction_to_ai_submit_persona_sweep.sbatch
#
# ---------------------------------------------------------------------------

#SBATCH --job-name=ia-sweep
#SBATCH --account=conf-icl-2026.09.25-albarass
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.out

set -uo pipefail

REPO=/ibex/user/habiam0b/autoGrader
cd "$REPO"
RESULTS="$REPO/introduction_to_ai_results"

: "${MODEL:?set MODEL=<hf id or local weight dir> via --export}"
: "${MODEL_TAG:?set MODEL_TAG=<short name used in the run tag> via --export}"
RID_PREFIX="${RID_PREFIX:-IA}"
TP="${TP:-1}"

PERSONAS=(neutral strict lenient rigorous exacting)
# PERSONAS_OVERRIDE replaces the persona list (space-separated, indexed by array
# task). It contains SPACES, so it must travel via `export` + --export=ALL, never
# inline in --export=... (the sbatch --export separator trap).
if [[ -n "${PERSONAS_OVERRIDE:-}" ]]; then
  read -r -a PERSONAS <<< "$PERSONAS_OVERRIDE"
fi
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
PERSONA="${PERSONAS[$TASK_ID]}"
# RID_BASE offsets the run number so several submissions can share one prefix.
RID_BASE="${RID_BASE:-0}"
RID=$(printf "%s%02d" "$RID_PREFIX" $((RID_BASE + TASK_ID + 1)))

# STAGE-2: gd is fixed at 0 -- no exam-guidelines document exists for this exam,
# and introduction_to_ai_grade_with_local.py refuses --with-guidelines outright.
BD="${BD:-1}"
BREAKDOWN_ARG=--with-breakdown
[[ "$BD" == "0" ]] && BREAKDOWN_ARG=--no-breakdown

# Remaining grid axes. All default to the published Stage 2 configuration, so
# existing submissions produce byte-identical tags and behaviour.
SOL="${SOL:-1}"
SOLUTION_ARG=--with-solution
[[ "$SOL" == "0" ]] && SOLUTION_ARG=--no-solution
RS="${RS:-0}"
REASONING_ARG=--no-reasoning
[[ "$RS" == "1" ]] && REASONING_ARG=--with-reasoning
TEMP="${TEMP:-0.0}"
RUNS="${RUNS:-1}"
FEWSHOT="${FEWSHOT:-0}"
T_STR="t$(printf '%02d' "${TEMP/./}")"
TAG="${RID}_m-${MODEL_TAG}_sol${SOL}_gd0_rs${RS}_bd${BD}_str-${PERSONA}_${T_STR}_n${RUNS}"
[[ "$FEWSHOT" != "0" ]] && TAG="${TAG}_fs${FEWSHOT}"
OUT="$RESULTS/results/${RID}__${TAG}.xlsx"
LOGF="$REPO/local_logs/${TAG}.log"
VARF="$RESULTS/variance/${TAG}.csv"
TRACKER="$RESULTS/ablation_runs.xlsx"


# STUDENTS=<subset> turns this into a throwaway smoke run: it writes outside the
# real results tree and outside the tracker on purpose, because the grader skips
# students that already have a score -- a smoke run writing into the real
# workbook would bake its rows (including any produced by the bug the smoke was
# meant to catch) permanently into the paper's run.
STUDENTS="${STUDENTS:-}"
if [[ -n "$STUDENTS" ]]; then
  TARGET_ARGS=(--students "$STUDENTS")
  OUT="$RESULTS/smoke/smoke_${RID}__${TAG}.xlsx"
  LOGF="$RESULTS/smoke/smoke_${TAG}.log"
  VARF="$RESULTS/smoke/smoke_${TAG}.csv"
  TRACKER="$RESULTS/smoke/smoke_tracker.xlsx"
  MODE_DESC="SMOKE students=$STUDENTS (NOT registered)"
else
  TARGET_ARGS=(--all)
  MODE_DESC="full sweep, all students"
fi

# SUBSET=<range> is a REAL, registered run over a subset of students -- not a
# smoke. The n>1 variance runs use it: their Stage 3 counterparts (L-E01/L-E02)
# sampled 50 students, not the whole cohort, and grading 1,038 x 5 samples would
# take ~90 h against a 24 h wall. Keeping the real paths is safe here precisely
# because a variance run writes to the variance CSV and never skips
# already-scored students, so none of the STUDENTS hazard above applies.
SUBSET="${SUBSET:-}"
if [[ -z "$STUDENTS" && -n "$SUBSET" ]]; then
  TARGET_ARGS=(--students "$SUBSET")
  MODE_DESC="registered subset, students=$SUBSET"
fi

# SAVE_IO=1 dumps every prompt/response (diagnostics only -- one file per call).
SAVEIO_ARGS=()
[[ -n "${SAVE_IO:-}" ]] && SAVEIO_ARGS=(--save-io)

mkdir -p "$RESULTS"/{results,variance} "$REPO"/{local_logs,slurm_logs}

# Serving stack. CONDA_ENV takes a name or a path; ENVBIN skips conda entirely.
# The published runs used vLLM 0.19.1.
ENVBIN="${ENVBIN:-/ibex/user/shiekhmf/envs/autograder-v19/bin}"
if [[ -n "${CONDA_ENV:-}" ]]; then
  source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  conda activate "$CONDA_ENV" || { echo "!! could not activate: $CONDA_ENV"; exit 1; }
  ENVBIN="$(dirname "$(which python)")"
fi
# Some conda envs ship a newer libstdc++ than the nodes do, and their bundled
# libzmq needs it; without it vLLM dies on an ImportError for GLIBCXX_3.4.xx.
# But this path must NOT be exported globally: these envs also ship libssl.so.3 /
# libcrypto.so.3, and putting them ahead of the system's breaks system binaries --
# `curl` in particular dies with "symbol lookup error: EVP_md2", which silently
# kills the readiness probe below and burns the whole WAIT_SECS on idle GPUs.
# So compute it here and apply it ONLY as a prefix on the two python invocations.
ENV_LD=""
[[ -d "$ENVBIN/../lib" ]] && ENV_LD="$(cd "$ENVBIN/../lib" && pwd)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
runpy() { if [[ -n "$ENV_LD" ]]; then LD_LIBRARY_PATH="$ENV_LD" "$ENVBIN/python" "$@"; else "$ENVBIN/python" "$@"; fi; }
# Weights are pre-warmed here; keep Hub access ON. Offline mode is NOT a fix --
# huggingface_hub runs a strict snapshot-completeness check offline and this
# cache lacks cosmetic files (README/LICENSE) for several models. Gated repos
# (meta-llama, google) 403 on anonymous config fetches instead, so those are
# served from plain local weight directories by path.
export HF_HOME="${HF_HOME:-/ibex/user/shiekhmf/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export VLLM_LOGGING_LEVEL=INFO
MAX_LEN="${MAX_LEN:-32768}"
MEM_UTIL="${MEM_UTIL:-0.92}"
DTYPE="${DTYPE:-bfloat16}"
# STAGE-2: TOKENIZER_MODE is passed through but does NOT fix Mistral -- vLLM
# picks the mistral-format tokenizer from the model itself even at
# tokenizer_mode=auto, and that tokenizer rejects any chat_template parameter.
# The grader handles it client-side by dropping chat_template_kwargs after the
# first rejection (a no-op for models with no thinking template).
TOKENIZER_ARGS=()
[[ -n "${TOKENIZER_MODE:-}" ]] && TOKENIZER_ARGS=(--tokenizer-mode "$TOKENIZER_MODE")

# Unique port per array task AND per array job: 8000+TASK_ID is not enough,
# since every sweep has a task 1 and ports are per-node, not per-job. Two jobs
# once met on one node and one graded 570 students against the other's model.
port_free() { ! (exec 3<>/dev/tcp/127.0.0.1/"$1") 2>/dev/null; }
PORT=$(( 8000 + (${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}} % 500) * 10 + TASK_ID ))
for _ in $(seq 1 50); do
  port_free "$PORT" && break
  PORT=$((PORT + 1))
done

STACK_VLLM=$(runpy -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo "?")
echo "=============================================================="
echo " run id      : $RID   (persona: $PERSONA)"
echo " mode        : $MODE_DESC"
echo " model       : $MODEL   (tag: $MODEL_TAG)"
echo " node        : $(hostname)   GPUs: ${CUDA_VISIBLE_DEVICES:-unset}   TP=$TP"
echo " port        : $PORT"
echo " output      : $OUT"
echo " stack       : vllm $STACK_VLLM   HF_HOME: $HF_HOME"
echo "=============================================================="

VLLM_LOG="$REPO/slurm_logs/vllm-${RID}-${SLURM_JOB_ID}.log"

# Parallelism topology. Plain TP is right for every dense model here, but the
# block-FP8 MoE (Qwen3-Coder-480B) MUST run TP=4 DP=2 EP=1: at TP=8 its MoE
# shard is 320 wide, which is not divisible by the block_n=128 FP8 quantization
# block, and the server dies on load. Total GPUs used = TP * DP, so the caller
# still requests 8. Mirrors serve_qwen_480B.sh.
PAR_ARGS=(--tensor-parallel-size "$TP")
[[ "${DP:-1}" -gt 1 ]]      && PAR_ARGS+=(--data-parallel-size "$DP")
[[ "${EP:-0}" == "1" ]]     && PAR_ARGS+=(--enable-expert-parallel)
[[ -n "${MAX_SEQS:-}" ]]    && PAR_ARGS+=(--max-num-seqs "$MAX_SEQS")
echo " parallelism : ${PAR_ARGS[*]}   (total GPUs = TP*DP = $(( TP * ${DP:-1} )))"

runpy -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port "$PORT" "${PAR_ARGS[@]}" \
  --max-model-len "$MAX_LEN" --gpu-memory-utilization "$MEM_UTIL" --dtype "$DTYPE" \
  "${TOKENIZER_ARGS[@]}" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
  echo "-- shutting down vLLM (pid $VLLM_PID)"
  kill "$VLLM_PID" 2>/dev/null || true
  for _ in $(seq 1 15); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 1; done
  pkill -9 -P "$VLLM_PID" 2>/dev/null || true
  kill -9 "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Ask the pid BEFORE the port, and make the port prove which model it serves:
# "something answers /v1/models" is not the question, "is it MY model" is.
WAIT_SECS="${WAIT_SECS:-3600}"
waited=0
while true; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "!! vLLM exited after ${waited}s without serving. Tail of $VLLM_LOG:"
    tail -30 "$VLLM_LOG"; exit 1
  fi
  resp=$(curl -sf --max-time 5 "http://localhost:${PORT}/v1/models" 2>/dev/null) || resp=""
  if [[ -n "$resp" ]]; then
    [[ "$resp" == *"\"${MODEL}\""* ]] && break
    echo "!! port ${PORT} serves a different model; refusing to grade against it."; exit 1
  fi
  if [[ "$waited" -ge "$WAIT_SECS" ]]; then
    echo "!! vLLM not ready after ${WAIT_SECS}s."; tail -30 "$VLLM_LOG"; exit 1
  fi
  sleep 10; waited=$((waited + 10))
done
echo "-- vLLM ready after ${waited}s (serving ${MODEL} on port ${PORT})"

runpy introduction_to_ai_grade_with_local.py \
  "${TARGET_ARGS[@]}" \
  "${SAVEIO_ARGS[@]}" \
  --model "$MODEL" \
  --api-base "http://localhost:${PORT}/v1" \
  --wait-for-server 600 \
  --max-output-tokens 8192 \
  "$SOLUTION_ARG" \
  "$REASONING_ARG" \
  "$BREAKDOWN_ARG" \
  --strictness "$PERSONA" \
  --temperature "$TEMP" \
  --runs "$RUNS" \
  --few-shot "$FEWSHOT" \
  --tag "$TAG" \
  --output-xlsx "$OUT" \
  --log-file "$LOGF" \
  --variance-csv "$VARF" \
  --tracker "$TRACKER"
STATUS=$?

echo "-- grader exited with status $STATUS"
exit $STATUS
