"""Run the fixed churn question set through the live LangGraph workflow.

This runner makes real Groq and executor calls. It records diagnostic workflow
results, not a correctness score; human review uses the generated review CSV.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import pandas as pd
from langchain_core.messages import HumanMessage

from evals.churn.fixture import DATA_DICTIONARY, build_fixture
from src.agents.graph import app as agent_app


CASES_PATH = Path(__file__).with_name("cases.json")
RESULTS_DIR = Path(__file__).with_name("results")


def _chart_summary(charts):
    summaries = []
    for chart in charts or []:
        layout = getattr(chart, "layout", None)
        xaxis = getattr(layout, "xaxis", None)
        yaxis = getattr(layout, "yaxis", None)
        summaries.append({
            "title": getattr(getattr(layout, "title", None), "text", None),
            "xaxis_title": getattr(getattr(xaxis, "title", None), "text", None),
            "yaxis_title": getattr(getattr(yaxis, "title", None), "text", None),
            "trace_types": [getattr(trace, "type", "unknown") for trace in chart.data],
        })
    return summaries


def _schema(frame):
    return (
        f"Columns: {list(frame.columns)}\n"
        f"Types: {frame.dtypes.astype(str).to_dict()}\n"
        f"Shape: {frame.shape}\n"
        f"Data dictionary:\n{DATA_DICTIONARY}"
    )


def _review_template(cases, output_path):
    import csv

    review_path = output_path.with_suffix(".review.csv")
    with review_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "case_id", "split", "intent_correct", "answer_correct",
            "evidence_supported", "chart_correct", "notes",
        ])
        writer.writeheader()
        for case in cases:
            writer.writerow({"case_id": case["id"], "split": case["split"]})
    return review_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("all", "dev", "holdout"), default="all")
    parser.add_argument("--output", type=Path, help="Optional path for JSON results.")
    args = parser.parse_args()

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if args.split != "all":
        cases = [case for case in cases if case["split"] == args.split]
    if not os.getenv("GROQ_API_KEY"):
        parser.error("GROQ_API_KEY must be configured; no secret value is printed.")
    if any(case["intent"] in {"ANALYZE", "MODIFY"} for case in cases):
        if not os.getenv("ANALYSIS_EXECUTOR_URL") or not os.getenv("ANALYSIS_EXECUTOR_TOKEN"):
            parser.error("ANALYSIS_EXECUTOR_URL and ANALYSIS_EXECUTOR_TOKEN must be configured.")
    frame = build_fixture()
    schema = _schema(frame)
    results = []

    for position, case in enumerate(cases, start=1):
        state = {
            "messages": [HumanMessage(content=case["question"])],
            "dataset_schema": schema,
            "dataframe": frame.copy(deep=True),
            "dataframe_blob": None,
            "plan": [], "code": "", "code_output": "", "explanation": "",
            "charts": [], "error": None, "retry_count": 0,
            "retry_history": [], "intent": "", "analysis_failed": False,
            "share_values": False, "evidence": {}, "verification": [],
            "clarification": "", "clarification_context": "",
        }
        started = time.perf_counter()
        try:
            result = agent_app.invoke(state)
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            returned_frame = result.get("dataframe")
            record = {
                "case_id": case["id"], "split": case["split"],
                "expected_intent": case["intent"], "observed_intent": result.get("intent"),
                "analysis_failed": bool(result.get("analysis_failed")),
                "error_category": result.get("error_category"),
                "retry_count": result.get("retry_count", 0),
                "elapsed_ms": elapsed_ms,
                "explanation": result.get("explanation", ""),
                "code_output": result.get("code_output", ""),
                "code": result.get("code", ""), "plan": result.get("plan", []),
                "charts": _chart_summary(result.get("charts", [])),
                "evidence": result.get("evidence", {}),
                "verification": result.get("verification", []),
                "dataframe_shape_before": list(frame.shape),
                "dataframe_shape_after": (list(returned_frame.shape)
                                           if isinstance(returned_frame, pd.DataFrame) else None),
                "dataframe_columns_after": (list(map(str, returned_frame.columns))
                                             if isinstance(returned_frame, pd.DataFrame) else None),
                "rubric": case["rubric"],
            }
        except Exception as exc:
            record = {
                "case_id": case["id"], "split": case["split"],
                "expected_intent": case["intent"], "observed_intent": None,
                "analysis_failed": True, "error_category": type(exc).__name__,
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
                "explanation": "Workflow raised an exception; inspect local logs.",
                "rubric": case["rubric"],
            }
        results.append(record)
        print(f"[{position}/{len(cases)}] {case['id']}: intent={record['observed_intent']} "
              f"failed={record['analysis_failed']} elapsed_ms={record['elapsed_ms']}")

    if args.output:
        output_path = args.output
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = RESULTS_DIR / f"churn-{args.split}-{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        "question_set_sha256": hashlib.sha256(CASES_PATH.read_bytes()).hexdigest(),
        "fixture": "synthetic-v1",
        "row_count": len(frame),
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    review_path = _review_template(cases, output_path)
    print(f"Saved diagnostic outputs: {output_path}")
    print(f"Saved manual-review template: {review_path}")
    print("No answer-accuracy score was inferred. Review each response against its rubric.")


if __name__ == "__main__":
    main()
