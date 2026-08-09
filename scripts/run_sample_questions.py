#!/usr/bin/env python3
"""
Runs all 5 sample questions from data/sample_questions.json, plus one
constructed case designed to fail verification on the first pass (to
demonstrate the retry path), and saves each structured response under
outputs/.

Usage:
    python scripts/run_sample_questions.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph import build_graph, build_models_and_index

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"


class ForceFailOnceGenerator:
    """Wraps a real generator but returns an evidence-free, ungrounded
    answer on its very first call, then delegates to the real generator
    for every call after that (including the automatic retry).

    Used only for the constructed "verification failure" demo case
    (Required Test Case 5), so the retry -> re-verify path can be shown
    live regardless of which local model is actually loaded.
    """

    def __init__(self, real_generator):
        self._real = real_generator
        self._first_call_used = False
        self.backend = real_generator.backend + "+forced-fail-once"

    def generate(self, prompt, system=None, max_new_tokens=400):
        if not self._first_call_used:
            self._first_call_used = True
            return "xyzzyplugh corvidae parsnip glockenspiel wobbulator quixotic ephemeral driftwood."
        return self._real.generate(prompt, system=system, max_new_tokens=max_new_tokens)


def load_sample_questions():
    data = json.loads((ROOT / "data" / "sample_questions.json").read_text())
    return data["questions"]


def main():
    OUTPUTS_DIR.mkdir(exist_ok=True)

    print("Loading models and building retrieval index ...\n")
    embedder, generator, index, metadata = build_models_and_index("data")
    print(json.dumps(metadata, indent=2))

    graph = build_graph(embedder=embedder, generator=generator, index=index)

    # Required Test Case 5: a case where the first generated answer fails
    # verification. This reuses ForceFailOnceGenerator so the retry path is
    # demonstrated reliably on camera regardless of which real local model
    # is loaded (a real small model *may* also fail verification on its
    # own, but that's not guaranteed / reproducible enough for a demo).
    forced_fail_graph = build_graph(
        embedder=embedder, generator=ForceFailOnceGenerator(generator), index=index
    )

    questions = load_sample_questions()
    questions.append(
        {
            "question_id": "Q-006-verification-retry-demo",
            "question": (
                "Our daily dashboard exports stopped after an Admin changed the "
                "workspace timezone. The schedule still shows active. What should we check?"
            ),
            "_use_forced_fail_graph": True,
        }
    )

    summary = []
    for q in questions:
        print("\n" + "#" * 70)
        print(f"# {q['question_id']}: {q['question']}")
        print("#" * 70)

        active_graph = forced_fail_graph if q.get("_use_forced_fail_graph") else graph
        t0 = time.time()
        result = active_graph.invoke({"question": q["question"]})
        elapsed = round(time.time() - t0, 3)

        print("NODE TRACE:", " -> ".join(result["node_trace"]))
        print("CLASSIFICATION:", result["final_response"]["classification"])
        print(f"LATENCY: {elapsed}s")

        out_path = OUTPUTS_DIR / f"{q['question_id']}.json"
        out_path.write_text(json.dumps(result["final_response"], indent=2))

        summary.append(
            {
                "question_id": q["question_id"],
                "classification": result["final_response"]["classification"],
                "node_trace": result["node_trace"],
                "latency_s": elapsed,
                "output_file": str(out_path.relative_to(ROOT)),
            }
        )

    (OUTPUTS_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n\nSaved outputs + run_summary.json under outputs/")


if __name__ == "__main__":
    main()
