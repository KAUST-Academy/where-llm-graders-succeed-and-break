# finetune/ — LoRA fine-tuning the open graders

Fine-tune the open Qwen graders, then measure whether they beat the un-tuned
baseline on a held-out split. Two independent supervision recipes:

| recipe | builder | orchestrator | target per (student, question) | supervision |
|--------|---------|--------------|--------------------------------|-------------|
| **marks** | `build_finetune_data.py` | `run_finetune.sh` | `{"score", "bonus"}` | **TA marks** (avg of TA1+TA2) |
| **bd** (breakdown-distill) | `build_breakdown_data.py` | `run_breakdown_finetune.sh` | full bd1 JSON: `{score, bonus, task_breakdown[…], reasoning}` | **Gemini D01** (distilled) |

Both reuse the **same seeded split** (`data_bd/` reads `data/split_meta.json`),
so the two experiments are compared on the identical held-out students.

Why two recipes:
- **marks** trains purely on the numbers the TAs actually recorded — per-question
  scores from `computer_vision_dataset/Practical_AI_exam_grades.xlsx`. The only real, human
  supervision available. TAs never recorded per-task marks, so there is no faithful
  per-task target here.
- **bd** distils Gemini **D01** (`gemini-3-flash-preview`, sol1/gd1/bd1/neutral),
  which grades each task by actually reading the code and tracks the TA-average
  (MAE 1.64) more tightly than the two TAs agree with each other (2.61). Its
  score+bonus+breakdown+reasoning are copied faithfully (not rescaled). This is a
  deliberate model-distillation arm — the per-task numbers come from a model, not
  a proportional guess.

Build the datasets (deterministic; re-running reproduces the same files):

```bash
PY=/ibex/user/shaebiyy/conda-environments/ali-env/bin/python
$PY finetune/build_finetune_data.py  --seed 42 --train-frac 0.8   # marks -> finetune/data
$PY finetune/build_breakdown_data.py                              # bd    -> finetune/data_bd (needs data/ first)
```

## What each recipe does (and deliberately doesn't)

- **marks — TA marks only.** Per-question target = average of TA 1 and TA 2. No
  model-generated text anywhere. The model learns to output the number, not a
  rationale. Prompt asks for `{score, bonus}` only.
- **bd — full grading JSON, distilled from D01.** Prompt is the real `bd1` grader
  prompt; target is D01's own JSON incl. `task_breakdown` + `reasoning`. Use this
  when you want the "distill the best model" comparison arm.
- **Dynamic, seed-fixed split.** Students shuffled with `--seed`, split
  `--train-frac` (default 0.8). Same seed → same split. `data_bd/` inherits it.
- **Same prompt as the real grader.** Both reuse the rubric / solution /
  guidelines assembly from `computer_vision_grade_with_local.py`.

## Models (one A100 each, sequential)

| key  | HF id                                | baseline neutral MAE |
|------|--------------------------------------|----------------------|
| 7b   | `Qwen/Qwen2.5-Coder-7B-Instruct`     | 5.68 (L-A07)         |
| 14b  | `Qwen/Qwen2.5-Coder-14B-Instruct`    | 3.51 (L-A14)         |
| 30b  | `Qwen/Qwen3-Coder-30B-A3B-Instruct`  | 4.50 (L-A30M)        |

QLoRA (4-bit base + LoRA) keeps each within a single 80 GB A100. On a 40 GB
A100, the 7B/14B are fine; for the 30B drop `MAX_SEQ_LEN` (e.g. 6144).

## Run it (on a GPU node, same as serve_qwen.sh)

Training and evaluation are **two separate steps**: the runners TRAIN ONLY, and
eval goes through vLLM (10-30x faster than HuggingFace `generate()`, and it
doesn't tie up the GPUs doing batch-size-1 decoding).

```bash
# --- 1. train (pick a recipe) ---
EPOCHS=2 GPUS=4 bash finetune/run_finetune.sh 7b            # marks -> adapters/7b
EPOCHS=2 GPUS=4 bash finetune/run_breakdown_finetune.sh 7b  # bd    -> adapters/7b-bd

# --- 2. evaluate (serves base + adapter off one vLLM server) ---
bash finetune/run_vllm_eval.sh 7b marks   # -> FT-A07-{base,lora}.xlsx
bash finetune/run_vllm_eval.sh 7b bd      # -> FT-A07bd-{base,lora}.xlsx
```

Result tags: `FT-<tag>-{base,lora}` (marks) and `FT-<tag>bd-{base,lora}` (bd),
e.g. `FT-A14bd-lora`.

Override knobs via env vars: `SEED=7 EPOCHS=2 bash finetune/run_finetune.sh 14b`.
Note `run_finetune.sh` defaults to `EPOCHS=3` while `run_breakdown_finetune.sh`
defaults to `EPOCHS=2` -- set `EPOCHS` explicitly to keep the two recipes
comparable.

### Or step by step

```bash
PY=/ibex/user/shaebiyy/conda-environments/ali-env/bin/python

# 1. build the split + JSONL (seed-fixed, dynamic)
$PY finetune/build_finetune_data.py --seed 42 --train-frac 0.8

# 2. train one adapter
$PY finetune/train_lora.py \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --data finetune/data/train.jsonl \
    --output-dir finetune/adapters/7b

# 3a. baseline: evaluate the BASE model on the held-out split
$PY finetune/eval_lora.py --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --adapter '' --tag FT-A07-base

# 3b. evaluate the FINE-TUNED model on the same split
$PY finetune/eval_lora.py --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --adapter finetune/adapters/7b --tag FT-A07-lora
```

Each eval prints `MAE vs TA-avg` and writes `finetune/results/<tag>.xlsx` in the standard
ablation shape. The FT runs stay out of the prompt-grid master comparison
(`analysis/computer_vision_run_analysis.py` reads only `computer_vision_results/results/`);
`finetune/pairwise_analysis.py` and the fine-tuning tables read them from `finetune/results/`.

## Fast eval via vLLM (recommended — 10–30× faster)

`eval_lora.py` uses HuggingFace `generate()` at batch size 1: no batching, and it
re-prefills the identical ~18k-char guidelines+rubric+solution prefix for all
~386 calls. The vLLM path fixes both (continuous batching + `--enable-prefix-caching`)
and reuses schema-guided decoding so responses always parse.

One server hosts the **base model and the LoRA adapter at once**, so both the
baseline and the fine-tune are graded without a restart:

```bash
# does everything: serve (base+adapter) -> eval base -> eval lora -> stop
bash finetune/run_vllm_eval.sh 7b bd        # or: 7b marks / 14b bd / ...

# point at a specific checkpoint if training was interrupted
bash finetune/run_vllm_eval.sh 7b marks finetune/adapters/7b/checkpoint-188
```

Manual, if you'd rather drive it yourself:

```bash
# terminal A -- serve base + adapter ('ft')
bash finetune/serve_lora.sh Qwen/Qwen2.5-Coder-7B-Instruct finetune/adapters/7b-bd

# terminal B -- grade with each
$PY finetune/eval_vllm.py --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --prompt-style breakdown --eval-students finetune/data_bd/eval_students.json \
    --tag FT-A07bd-base
$PY finetune/eval_vllm.py --model ft \
    --prompt-style breakdown --eval-students finetune/data_bd/eval_students.json \
    --tag FT-A07bd-lora
```

Knobs: `CONCURRENCY=32` (in-flight requests — this is what lets vLLM batch;
a sequential client gets no speedup), `TP=1` (tensor-parallel GPUs), `PORT=8000`.

`--prompt-style` **must** match how the adapter was trained (`marks` ↔
`finetune/data`, `breakdown` ↔ `finetune/data_bd`) — the prompts are built by the
same functions the data builders use, so they stay byte-identical.

> **MoE caveat:** vLLM's LoRA support doesn't cover MoE expert layers on some
> architectures. Dense Qwen2.5-Coder 7B/14B are fine; if Qwen3-Coder-30B-A3B is
> rejected at load, fall back to `eval_lora.py` for that model.

## Reading the result

The headline is **base MAE vs fine-tuned MAE on the identical held-out split**
(printed by `eval_lora.py`). Because the split is the same for both, the delta
is the fine-tuning lift. Targets to keep in mind from the existing analysis:
Gemini D01 = 1.64, human floor = 2.61.

## Outputs

```
finetune/data/      train.jsonl eval.jsonl eval_students.json split_meta.json   # marks variant
finetune/data_bd/   (same four files)                                           # bd variant
finetune/adapters/<key>/       # marks-trained LoRA adapter per model
finetune/adapters/<key>-bd/    # bd-trained LoRA adapter per model
finetune/results/FT-<tag>-{base,lora}.xlsx      # marks runs
finetune/results/FT-<tag>bd-{base,lora}.xlsx    # bd runs
```

To eval a bd adapter by hand, match the prompt style and split dir:

```bash
$PY finetune/eval_lora.py --model Qwen/Qwen2.5-Coder-14B-Instruct \
    --adapter finetune/adapters/14b-bd --prompt-style breakdown \
    --eval-students finetune/data_bd/eval_students.json --tag FT-A14bd-lora
```

## Notes

- If flash-attn is not installed, scripts default to `--attn sdpa`. Pass
  `--attn flash_attention_2` if you install it.
- Long notebooks are head+tail truncated to `--max-student-chars` (24000 ≈ 6k
  tokens) so prompts fit the train context; eval uses the same cap for
  consistency.
