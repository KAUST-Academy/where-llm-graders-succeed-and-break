# Where LLM Graders Succeed and Break: Evidence from Two Computer-Science Exams

Code, data and results for the paper. Everything the paper's numbers, tables
and figures are computed from is here, laid out so that the scripts run as
they are.

| path | contents |
|---|---|
| `main.tex`, `sections/`, `media/`, `references.bib`, `main.bbl`, `PRIMEarxiv.sty` | the paper's LaTeX source |
| `computer_vision_dataset/`, `introduction_to_ai_dataset/` | the two exams: student notebooks, both graders' marks, rubrics, reference solutions (each has its own `README.md`) |
| `computer_vision_results/`, `introduction_to_ai_results/` | one workbook per grader configuration (`results/`), per-call variance logs (`variance/`), and the run tracker (`ablation_runs.xlsx`). `computer_vision_results/results/quarantine/` holds two DeepSeek-Coder-V2-236B runs that failed and are not among the paper's 171 configurations |
| `prompt_baselines/results/` | the prompt-only baselines |
| `closed_runs/` | batch metadata and token cost of the OpenAI and Anthropic runs, read by the closed-model tables |
| `finetune/` | the LoRA fine-tuning pipeline, its held-out splits, and its evaluation workbooks (`finetune/results/`) |
| `analysis/` | the analysis, table and figure scripts; `analysis/checks/` holds the re-computations the appendix names |
| `*_grade_with_local.py`, `*_grade_with_gemini.py`, `closed_batch_grade.py`, `prompt_baselines_grade.py`, `*.sbatch`, `serve_*.sh` | the graders and their launchers |
| `*_build_few_shot_examples.py`, `*_download_dataset.py`, `merge_gemini_shards.py` | few-shot bank builders, Hub download scripts, and the merger for sharded Gemini runs |

## Regenerating the paper

Tested with Python 3.12.

```bash
pip install pandas numpy scipy matplotlib openpyxl

python analysis/computer_vision_run_analysis.py       # analysis/computer_vision_master_comparison.csv
python analysis/introduction_to_ai_run_analysis.py    # analysis/introduction_to_ai_master_comparison.csv

python analysis/computer_vision_make_appendix_table.py     # sections/appendix_runs_table.tex
python analysis/introduction_to_ai_make_appendix_table.py  # sections/introduction_to_ai_appendix_runs_table.tex
python analysis/closed_vendor_table.py                     # sections/closed_vendors_table.tex
python analysis/computer_vision_make_paper_tables.py       # prints tab:brittle, tab:wording, tab:mechanism
python analysis/introduction_to_ai_make_paper_tables.py    # prints tab:replication, tab:repmech

python analysis/computer_vision_make_paper_figures.py      # media/
python analysis/introduction_to_ai_make_paper_figures.py
python analysis/make_cross_exam_figure.py

latexmk -pdf main.tex
```

The two run-analysis scripts also write Markdown reports to `analysis/` and four
figures the paper does not use to `media/` (all ignored by git). The
`*_make_paper_tables.py` scripts print LaTeX table bodies rather than writing
files. The regenerated figures differ from the committed ones only in their
embedded creation date. Each script in `analysis/checks/` prints the numbers of
the appendix table or paragraph that names it.

## Re-running the experiments

The grading and fine-tuning runs need GPUs or API access.

- **Graders.** Each row of a run tracker (`*_results/ablation_runs.xlsx`)
  records, in its `command` column, the exact grader invocation that produced
  that run. Give a re-run its own `--tag` and `--output-xlsx`: a grader resumes
  into an existing workbook and skips every question already scored. The
  Gemini graders read a Vertex AI service-account key from `vertex_key.json` at
  the repository root; `closed_batch_grade.py` reads `GPT=` and `CLAUDE=` API keys
  from `.env`. Both files are ignored by git.
- **Fine-tuning.** `finetune/README.md` describes the single-exam pipeline
  (the `FT-*` workbooks). The cross-exam and pooled runs behind the paper's
  fine-tuning tables (the `X-*` workbooks) are reproduced by the commands in
  section 17 of `finetune/finetune_findings.md`. The training data is rebuilt
  by `finetune/build_finetune_data.py` and `finetune/build_breakdown_data.py`;
  the adapters themselves are not included and must be retrained.

## Licence

- **Code** (the graders, launchers, fine-tuning pipeline and analysis scripts):
  MIT, see `LICENSE`.
- **Data** (`computer_vision_dataset/`, `introduction_to_ai_dataset/`, and the
  result workbooks in `*_results/`, `finetune/results/`, `prompt_baselines/results/`
  and `closed_runs/`): CC BY-NC 4.0, see `LICENSE-DATA`. The student work was
  contributed with the course instructors' permission for research on automated
  grading; please do not attempt to re-identify students or graders.
- **Paper source** (`main.tex`, `sections/`, `media/`): CC BY 4.0, see
  `LICENSE-PAPER`. `PRIMEarxiv.sty` is adapted from George Kour's arxiv-style
  and keeps its MIT notice.

## Citation

```bibtex
@misc{habibullah2026llmgraderssucceedbreak,
      title={Where LLM Graders Succeed and Break: Evidence from Two Computer-Science Exams}, 
      author={Ali Habibullah and Yazan Alshoibi and Mohammad Alshiekh and Salman Khan and Naeemullah Khan},
      year={2026},
      eprint={2609.29333},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.29333}, 
}
```
