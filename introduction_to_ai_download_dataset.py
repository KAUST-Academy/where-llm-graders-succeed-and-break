#!/usr/bin/env python3
"""Download the Introduction to AI (Stage 2) practical exam data.

One of the two exam datasets in this project; the Computer Vision (Stage 3)
data is fetched the same way by its own downloader into
`computer_vision_dataset/`.

    pip install huggingface_hub
    python introduction_to_ai_download_dataset.py

Pulls https://huggingface.co/datasets/KAUSTAcademy/AutoGrader_Introduction_to_AI
into an `introduction_to_ai_dataset/` folder next to this script — about 31 MB
across 3,025 files:

    introduction_to_ai_dataset/
      Practical_AI_exam_grades.csv      ground truth: two TA grades per student
      Solutions/                        reference solution per question
      submissions_extracted/1..1038/    student submissions, Q1.ipynb ... Q3.ipynb
      stage_2_rubrics_ta.md             the rubric the TAs graded against
      README.md                         dataset card

The dataset is currently PRIVATE, so this needs a token with read access:

    hf auth login          # or: export HF_TOKEN=<token>

Not every student submitted every question — an absent `Q<n>.ipynb` means that
question was not handed in, and the graders score it 0. Expect roughly 3,018
notebooks across 1,038 students rather than 3 x 1,038.

Safe to re-run: files already downloaded are skipped, so an interrupted
download picks up where it left off.
"""
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import (GatedRepoError, RepositoryNotFoundError)
except ImportError:
    sys.exit("This script needs huggingface_hub:\n\n    pip install huggingface_hub")

REPO_ID = "KAUSTAcademy/AutoGrader_Introduction_to_AI"
DEST = Path(__file__).resolve().parent / "introduction_to_ai_dataset"

AUTH_HINT = (
    f"\nCould not reach {REPO_ID}.\n"
    "This dataset is private, and the Hub answers 404 to unauthenticated\n"
    "requests for a private repo — so a missing token looks exactly like a\n"
    "wrong name. Check both:\n"
    "  1. authenticate:  hf auth login    (or export HF_TOKEN=<token>)\n"
    "  2. confirm you have access to the repository."
)


def main() -> int:
    print(f"Downloading {REPO_ID}")
    print(f"        to {DEST}")
    print("~31 MB, 3,025 files. This takes a couple of minutes.\n")

    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(DEST),
            max_workers=8,
        )
    except (RepositoryNotFoundError, GatedRepoError):
        sys.exit(AUTH_HINT)
    except Exception as e:  # noqa: BLE001
        # A bad or expired token surfaces as a plain HTTP error rather than one
        # of the typed exceptions above; say so instead of dumping a traceback.
        if any(code in str(e) for code in ("401", "403")):
            sys.exit(AUTH_HINT + f"\n\n(server said: {str(e).splitlines()[0][:120]})")
        raise

    sub = DEST / "submissions_extracted"
    students = sorted((d for d in sub.iterdir() if d.is_dir() and d.name.isdigit()),
                      key=lambda d: int(d.name))
    notebooks = sum(len(list(d.glob("Q*.ipynb"))) for d in students)
    missing = 3 * len(students) - notebooks
    print(f"\nDone. {len(students)} students, {notebooks} submission notebooks "
          f"({missing} question(s) not handed in).")
    print(f"Ground truth: {DEST / 'Practical_AI_exam_grades.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
