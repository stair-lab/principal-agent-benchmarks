"""
Materialize an items.jsonl file with prompts + choices for every (task, doc_id)
pair in welfare.csv, using lm-evaluation-harness as the source of truth.

Run this once before `make refresh-judge` to produce the input the LLM-judge
needs. The OLMES item set is not part of this paper's contributions and is
fetched from upstream (https://github.com/allenai/olmes) on demand.

Reads:
  data/welfare.csv          — provides the (task, doc_id) pairs to fetch.

Writes:
  data/items.jsonl          — one row per item: {task, doc_id, prompt, choices}.

Run:
  pip install lm-eval
  python -m analysis.fetch_olmes_items
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    try:
        from lm_eval import tasks as lm_eval_tasks
    except ImportError:
        raise SystemExit(
            "lm-evaluation-harness not installed.\n"
            "Run: pip install 'lm-eval[api]>=0.4.5'\n"
            "Then: python -m analysis.fetch_olmes_items"
        )

    welfare = pd.read_csv(DATA / "welfare.csv")
    by_task: dict[str, list[int]] = defaultdict(list)
    for _, r in welfare.iterrows():
        by_task[r["task"]].append(int(r["doc_id"]))

    out_path = DATA / "items.jsonl"
    n_total = sum(len(v) for v in by_task.values())
    print(f"Fetching {n_total} items across {len(by_task)} tasks via lm-evaluation-harness...")

    task_manager = lm_eval_tasks.TaskManager()
    n_written = 0
    with out_path.open("w") as out_f:
        for task_name, doc_ids in by_task.items():
            try:
                task_dict = lm_eval_tasks.get_task_dict([task_name], task_manager)
            except Exception as e:
                print(f"  [skip] {task_name}: {e}", file=sys.stderr)
                continue
            task = task_dict[task_name]
            docs = list(task.test_docs() if task.has_test_docs()
                        else task.validation_docs() if task.has_validation_docs()
                        else task.training_docs())
            wanted = set(doc_ids)
            for i, doc in enumerate(docs):
                if i not in wanted:
                    continue
                # Heuristic prompt extraction — works for OLMES MCQ tasks.
                prompt = doc.get("question") or doc.get("query") or doc.get("ctx") or ""
                choices = doc.get("choices") or doc.get("options") or []
                if isinstance(choices, dict):
                    choices = choices.get("text", [])
                row = {
                    "task": task_name,
                    "doc_id": i,
                    "prompt": prompt,
                    "choices": list(choices),
                }
                out_f.write(json.dumps(row) + "\n")
                n_written += 1
            print(f"  {task_name}: wrote {sum(1 for d in doc_ids if d < len(docs))} / "
                  f"{len(doc_ids)} requested doc_ids")

    print(f"\nwrote {out_path}  ({n_written} rows)")


if __name__ == "__main__":
    main()
