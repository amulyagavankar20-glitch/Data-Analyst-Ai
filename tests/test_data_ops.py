import io

import pandas as pd
import pytest

from src.data_ops import left_join_tables


def test_bounded_duckdb_join_preserves_rows_and_returns_parquet():
    left = pd.DataFrame({"customer_id": [1, 2, 1], "spend": [10, 20, 30]})
    right = pd.DataFrame({"customer_id": [1, 2], "region": ["east", "west"]})
    joined, blob = left_join_tables(left, right, "customer_id", "customer_id")
    assert len(joined) == len(left)
    assert joined["region"].tolist() == ["east", "west", "east"]
    pd.testing.assert_frame_equal(pd.read_parquet(io.BytesIO(blob)), joined)


def test_join_rejects_duplicate_right_keys_and_oversized_output():
    left = pd.DataFrame({"id": [1], "x": [1]})
    duplicate = pd.DataFrame({"id": [1, 1], "y": [2, 3]})
    with pytest.raises(ValueError, match="unique"):
        left_join_tables(left, duplicate, "id", "id")
    right = pd.DataFrame({"id": [1], "y": [2]})
    with pytest.raises(ValueError, match="column limit"):
        left_join_tables(left, right, "id", "id", max_columns=2)
