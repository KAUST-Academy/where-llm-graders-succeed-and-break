#!/usr/bin/env python3
"""Download the Computer Vision (Stage 3) practical exam data.

One of the two exam datasets in this project; the Introduction to AI (Stage 2)
data is fetched the same way by its own downloader into
`introduction_to_ai_dataset/`.

    pip install huggingface_hub
    python computer_vision_download_dataset.py

Pulls https://huggingface.co/datasets/KAUSTAcademy/AutoGrader_Computer_Vision into a `computer_vision_dataset/`
folder next to this script — about 19 MB across 1,895 files:

    computer_vision_dataset/
      Practical_AI_exam_grades.xlsx     ground truth: two TA scores per student
      Questions/                        the four exam questions
      Solutions/                        reference solutions
      submissions_extracted/1..570/     student submissions, Q1.ipynb ... Q4.ipynb
      stage_3_rubrics_ta.md             the rubric the TAs graded against
      exam_guidelines_student.md        exam policy shown to students
      few_shot_examples.json            worked examples for few-shot prompting

Safe to re-run: files already downloaded are skipped, so an interrupted
download picks up where it left off.
"""
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError
except ImportError:
    sys.exit("This script needs huggingface_hub:\n\n    pip install huggingface_hub")

REPO_ID = "KAUSTAcademy/AutoGrader_Computer_Vision"
DEST = Path(__file__).resolve().parent / "computer_vision_dataset"


def main() -> int:
    print(f"Downloading {REPO_ID}")
    print(f"        to {DEST}")
    print("~19 MB, 1,895 files. This takes a couple of minutes.\n")

    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(DEST),
            max_workers=8,
        )
    except RepositoryNotFoundError:
        sys.exit(
            f"\nCould not reach {REPO_ID}.\n"
            "Either the name is wrong or the dataset is not public yet.\n"
            "If you were given access, log in first:  hf auth login"
        )

    students = sorted(
        (d for d in (DEST / "submissions_extracted").iterdir() if d.is_dir()),
        key=lambda d: int(d.name) if d.name.isdigit() else 0,
    )
    notebooks = sum(len(list(d.glob("Q*.ipynb"))) for d in students)
    print(f"\nDone. {len(students)} students, {notebooks} submission notebooks.")
    print(f"Ground truth: {DEST / 'Practical_AI_exam_grades.xlsx'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
