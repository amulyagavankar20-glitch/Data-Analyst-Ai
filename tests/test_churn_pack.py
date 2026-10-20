import json
from pathlib import Path

from evals.churn.fixture import build_fixture
from src.agents.churn_pack import deterministic_churn_code
from src.agents.graph import validate_code


def test_dev_churn_pack_cases_get_vetted_templates():
    cases = json.loads(Path("evals/churn/cases.json").read_text(encoding="utf-8"))
    frame = build_fixture()
    generated = {}
    for case in (item for item in cases if item["split"] == "dev"):
        code = deterministic_churn_code(case["question"], frame)
        assert code is not None, case["id"]
        assert validate_code(code) is None, case["id"]
        generated[case["id"]] = code

    assert "churn.class_balance" in generated["C02"]
    assert "row['percent']:.2f" in generated["C02"]
    assert "xaxis_title='Churn Status'" in generated["C03"]
    assert "churn.segment_lift" in generated["C07"]
    assert "stats.grouped_mean" in generated["C08"]
    assert "churn.rate_by_bins" in generated["C11"]
    assert "stats.compare_groups" in generated["C12"]
    assert "verify.duplicates" in generated["C14"]
