#!/usr/bin/env python3
"""
CLI entrypoint to run replays and probes evaluation.

Usage:
    uv run python scripts/run_eval.py --url http://localhost:8000
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import evaluate_trace, TRACES_DIR
from eval.probes import run_all_probes

async def main():
    parser = argparse.ArgumentParser(description="SHL Recommender Evaluation Runner")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Target API base URL (default: http://localhost:8000)"
    )
    args = parser.parse_args()

    print(f"\nEvaluating target service: {args.url}")
    print("=" * 60)

    # 1. Run trace evaluation replays
    trace_files = sorted(TRACES_DIR.glob("*.md"))
    if not trace_files:
        print("Error: No markdown trace files found in eval/traces/")
        sys.exit(1)

    print(f"Found {len(trace_files)} traces. Replaying turns...")
    
    recalls = []
    trace_results = []
    
    for trace_file in trace_files:
        recall = await evaluate_trace(trace_file, args.url)
        recalls.append(recall)
        trace_results.append((trace_file.name, recall))
        print(f"  {trace_file.name}: Recall@10 = {recall:.2%}")

    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    print("-" * 60)
    print(f"Mean Recall@10: {mean_recall:.2%}")
    print("=" * 60)

    # 2. Run diagnostic behavior probes
    print("Running diagnostic behavior probes...")
    probe_results = await run_all_probes()
    
    passed_probes = 0
    for name, (passed, reason) in probe_results.items():
        status_str = "PASS" if passed else "FAIL"
        print(f"  [{status_str}] {name}: {reason}")
        if passed:
            passed_probes += 1

    probe_pass_rate = passed_probes / len(probe_results) if probe_results else 0.0
    print("-" * 60)
    print(f"Probes Pass Rate: {probe_pass_rate:.2%}")
    print("=" * 60)

    # 3. Print final summaries
    print("\nEVALUATION SUMMARY")
    print("-" * 35)
    print(f"Mean Recall@10:   {mean_recall:.2%}")
    print(f"Probe Pass Rate:  {probe_pass_rate:.2%}")
    print(f"Hard Eval Status: {'PASSED' if (mean_recall >= 0.70 and probe_pass_rate == 1.0) else 'COMPLETED'}")
    print("-" * 35 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
