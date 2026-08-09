#!/usr/bin/env python3
"""
CLI entry point.

Usage:
    python scripts/run_cli.py "Can a read-only Viewer create an API credential?"

Prints the node trace, a human-readable answer, and the full structured
JSON response (matching data/output_schema.json).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph import build_graph, build_models_and_index


def main():
    parser = argparse.ArgumentParser(description="Ask the OrbitDesk support agent a question.")
    parser.add_argument("question", type=str, help="Natural-language support question")
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    print(f"Loading models and building retrieval index from '{args.data_dir}' ...\n")
    embedder, generator, index, metadata = build_models_and_index(args.data_dir)
    print("Model/index metadata:")
    print(json.dumps(metadata, indent=2))
    print()

    graph = build_graph(embedder=embedder, generator=generator, index=index)

    result = graph.invoke({"question": args.question})

    print("=" * 70)
    print("NODE TRACE:", " -> ".join(result["node_trace"]))
    print("=" * 70)
    print("\nANSWER:\n" + result["final_response"]["answer"])
    print("\nSTRUCTURED RESPONSE:")
    print(json.dumps(result["final_response"], indent=2))


if __name__ == "__main__":
    main()
