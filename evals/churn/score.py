"""Summarize routing and manually reviewed answer quality for a run."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


def wilson_interval(successes: int, total: int, z: float = 1.96):
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _manual_rate(rows, field):
    reviewed = [row[field].strip().lower() for row in rows
                if row.get(field, "").strip().lower() in {"yes", "no"}]
    if not reviewed:
        return None
    correct = sum(value == "yes" for value in reviewed)
    interval = wilson_interval(correct, len(reviewed))
    return {"correct": correct, "reviewed": len(reviewed),
            "rate": correct / len(reviewed), "wilson_95": interval}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="JSON file produced by run.py")
    parser.add_argument("review", type=Path, help="Completed .review.csv file")
    args = parser.parse_args()

    results = json.loads(args.results.read_text(encoding="utf-8"))["results"]
    with args.review.open(newline="", encoding="utf-8") as stream:
        reviews = {row["case_id"]: row for row in csv.DictReader(stream)}

    total = len(results)
    routed = sum(item.get("observed_intent") == item["expected_intent"] for item in results)
    analysis_cases = [item for item in results if item["expected_intent"] in {"ANALYZE", "MODIFY"}]
    completed = sum(not item.get("analysis_failed", True) for item in analysis_cases)
    latencies = [item["elapsed_ms"] for item in results if isinstance(item.get("elapsed_ms"), int)]
    merged = [{**item, **reviews.get(item["case_id"], {})} for item in results]

    report = {
        "cases": total,
        "intent_correct": {"correct": routed, "total": total,
                           "rate": routed / total if total else None},
        "analysis_or_modify_completed": {"completed": completed, "total": len(analysis_cases),
                                          "rate": completed / len(analysis_cases) if analysis_cases else None},
        "latency_ms_median": median(latencies) if latencies else None,
        "manual_answer_correctness": _manual_rate(merged, "answer_correct"),
        "manual_evidence_support": _manual_rate(merged, "evidence_supported"),
        "manual_chart_correctness": _manual_rate(merged, "chart_correct"),
        "note": "Manual correctness rates use Wilson 95% intervals; workflow completion is not answer accuracy.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
