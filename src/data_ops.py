"""Bounded local data operations used by the UI."""
from io import BytesIO

import duckdb
import pandas as pd


def left_join_tables(left: pd.DataFrame, right: pd.DataFrame, left_key: str,
                     right_key: str, *, max_rows: int = 500_000,
                     max_columns: int = 200, max_parquet_bytes: int = 16 * 1024 * 1024
                     ) -> tuple[pd.DataFrame, bytes]:
    """Perform a many-to-one left join and enforce output limits before commit."""
    if left_key not in left.columns or right_key not in right.columns:
        raise ValueError("Join keys must exist in their selected tables.")
    if right[right_key].dropna().duplicated().any():
        raise ValueError("The right table's non-empty join key must be unique to keep the result bounded.")
    output_column_count = len(left.columns) + len(right.columns) - 1
    if output_column_count > max_columns:
        raise ValueError(f"Joined dataset would exceed the {max_columns}-column limit.")

    def quote(identifier):
        return '"' + str(identifier).replace('"', '""') + '"'

    connection = duckdb.connect(":memory:")
    try:
        connection.register("left_table", left)
        connection.register("right_table", right)
        right_columns = [f"r.{quote(column)} AS {quote(column)}"
                         for column in right.columns if column != right_key]
        selection = "l.*" + (", " + ", ".join(right_columns) if right_columns else "")
        joined = connection.execute(
            f"SELECT {selection} FROM left_table l LEFT JOIN right_table r "
            f"ON l.{quote(left_key)} = r.{quote(right_key)}"
        ).df()
    finally:
        connection.close()
    if len(joined) > max_rows:
        raise ValueError(f"Joined dataset exceeds the {max_rows:,}-row limit.")
    buffer = BytesIO()
    joined.to_parquet(buffer, index=None, compression="zstd")
    if buffer.tell() > max_parquet_bytes:
        raise ValueError("Joined dataset exceeds the configured Parquet size limit.")
    return joined, buffer.getvalue()
