#!/bin/bash
# Item 3 (LLM_CHECKLIST.md): prompt-only baselines under the strict persona.
# One job = one open model: start its vLLM server (the paper's stack, vLLM
# 0.19.1 from the autograder-v19 env), then run prompt_baselines_grade.py in
# both modes (decomposed, then arbitrated) on CV students 1-100, Q1-Q3.
#
#   sbatch --gres=gpu:a100:2 --export=ALL,MODEL=Qwen/Qwen2.5-Coder-32B-Instruct,MODEL_TAG=qwen2.5-coder-32b,IDX=01,TP=2 prompt_baselines_submit.sbatch
#   sbatch --gres=gpu:a100:2 --export=ALL,MODEL=zai-org/GLM-4-32B-0414,MODEL_TAG=glm-4-32b,IDX=02,TP=2 prompt_baselines_submit.sbatch
#   sbatch --gres=gpu:a100:1 --export=ALL,MODEL=mistralai/Mistral-Small-24B-Instruct-2501,MODEL_TAG=mistral-small-24b,IDX=03,TP=1 prompt_baselines_submit.sbatch
#   sbatch --gres=gpu:a100:1 --export=ALL,MODEL=meta-llama/Llama-3.1-8B-Instruct,MODEL_TAG=llama-3.1-8b,IDX=04,TP=1,LOCAL_SNAPSHOT=1 prompt_baselines_submit.sbatch
#
# LOCAL_SNAPSHOT=1 serves a gated repo from its cached snapshot directory
# (anonymous config fetches 403 on gated repos), keeping the HF id as the
# served model name. Run ids: PB-D<IDX> (decomposed) and PB-A<IDX> (arbitrated).

#SBATCH --job-name=pb-strict
#SBATCH --account=conf-icl-2026.09.25-albarass
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

set -uo pipefail
REPO="${REPO:-/ibex/user/habiam0b/autoGrader}"
cd "$REPO"
: "${MODEL:?}"; : "${MODEL_TAG:?}"; : "${IDX:?}"
TP="${TP:-1}"
STUDENTS="${STUDENTS:-1-100}"
CONC="${CONC:-4}"
MODES="${MODES:-decomposed arbitrated}"

ENVBIN="${ENVBIN:-/ibex/user/shiekhmf/envs/autograder-v19/bin}"
ENV_LD="$(cd "$ENVBIN/../lib" && pwd)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
runpy() { LD_LIBRARY_PATH="$ENV_LD" "$ENVBIN/python" "$@"; }
# The weights live in the co-author's readable cache. Do NOT inherit the submitting
# shell's HF_HOME (--export=ALL carries it): job 52130165 looked in the wrong cache and
# died, and the sibling jobs started downloading 60 GB checkpoints. PB_HF_HOME overrides.
export HF_HOME="${PB_HF_HOME:-/ibex/user/shiekhmf/hf_cache}"
export HF_HUB_OFFLINE=0
export VLLM_LOGGING_LEVEL=INFO
mkdir -p slurm_logs prompt_baselines

SERVE_PATH="$MODEL"
if [[ "${LOCAL_SNAPSHOT:-0}" == "1" ]]; then
  SERVE_PATH=$(ls -d "$HF_HOME/hub/models--${MODEL//\//--}/snapshots/"*/ | head -1)
  [[ -d "$SERVE_PATH" ]] || { echo "!! no cached snapshot for $MODEL"; exit 1; }
fi

port_free() { ! (exec 3<>/dev/tcp/127.0.0.1/"$1") 2>/dev/null; }
PORT=$(( 8000 + (${SLURM_JOB_ID:-0} % 500) * 10 ))
for _ in $(seq 1 50); do port_free "$PORT" && break; PORT=$((PORT + 1)); done

echo "== $(date) job $SLURM_JOB_ID on $(hostname): $MODEL (serve path $SERVE_PATH) TP=$TP port=$PORT students=$STUDENTS modes=$MODES"
echo "   vllm $(runpy -c 'import vllm; print(vllm.__version__)' 2>/dev/null) | git $(git rev-parse --short HEAD)"

VLLM_LOG="slurm_logs/vllm-PB${IDX}-${SLURM_JOB_ID}.log"
runpy -m vllm.entrypoints.openai.api_server --model "$SERVE_PATH" --served-model-name "$MODEL" \
  --host 0.0.0.0 --port "$PORT" --tensor-parallel-size "$TP" --max-model-len 32768 --max-num-seqs 64 \
  --dtype bfloat16 --gpu-memory-utilization 0.92 --enable-prefix-caching --trust-remote-code > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
cleanup() { kill "$VLLM_PID" 2>/dev/null || true; for _ in $(seq 1 15); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 1; done; pkill -9 -P "$VLLM_PID" 2>/dev/null || true; kill -9 "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

waited=0
while true; do
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "!! vLLM died after ${waited}s"; tail -30 "$VLLM_LOG"; exit 1; }
  resp=$(curl -sf --max-time 5 "http://localhost:${PORT}/v1/models" 2>/dev/null)
  if [[ -n "$resp" ]]; then
    [[ "$resp" == *"\"${MODEL}\""* ]] && break
    echo "!! port $PORT serves something else: ${resp:0:200}"; exit 1
  fi
  [[ "$waited" -ge 3600 ]] && { echo "!! vLLM not ready after ${waited}s"; tail -30 "$VLLM_LOG"; exit 1; }
  sleep 10; waited=$((waited + 10))
done
echo "-- vLLM ready after ${waited}s"

STATUS=0
for MODE in $MODES; do
  case "$MODE" in decomposed) RID="PB-D${IDX}";; arbitrated) RID="PB-A${IDX}";; *) echo "bad mode $MODE"; exit 2;; esac
  echo "== $(date) $RID ($MODE) start"
  runpy prompt_baselines_grade.py grade --mode "$MODE" --run-id "$RID" --model "$MODEL" --model-tag "$MODEL_TAG" \
    --api-base "http://localhost:${PORT}/v1" --students "$STUDENTS" --questions 1,2,3 --persona strict \
    --max-output-tokens 4096 --concurrency "$CONC" 2>&1 | grep -vE "HTTP Request" | tail -400
  rc=${PIPESTATUS[0]}; echo "== $(date) $RID exit $rc"; [[ $rc -ne 0 ]] && STATUS=$rc
done
exit $STATUS
