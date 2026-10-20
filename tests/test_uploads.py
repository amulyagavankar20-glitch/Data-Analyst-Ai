"""Format-specific upload parsing checks."""
import io
import pandas as pd
import pytest
import xlwt

from src.uploads import parse_uploaded_table


@pytest.mark.parametrize("name,raw", [
    ("sample.csv", b"customer_id,value\n00123,4\n00007,5\n"),
    ("sample.tsv", b"customer_id\tvalue\n00123\t4\n00007\t5\n"),
    ("sample.json", b'[{"id":"00123","value":4}]'),
    ("sample.json", b'{"id":["00123"],"value":[4]}'),
    ("sample.jsonl", b'{"id":"00123","value":4}\n'),
    ("sample.ndjson", b'{"id":"00123","value":4}\n'),
])
def test_reads_text_tabular_formats_without_losing_identifier_text(name, raw):
    frame = parse_uploaded_table(raw, name)
    assert frame.iloc[0]["id" if name.endswith(("json", "jsonl", "ndjson")) else "customer_id"] == "00123"


def test_reads_parquet_and_xlsx():
    source = pd.DataFrame({"id": ["00123"], "value": [4]})
    parquet = io.BytesIO()
    source.to_parquet(parquet, index=False)
    pd.testing.assert_frame_equal(parse_uploaded_table(parquet.getvalue(), "sample.parquet"), source)

    workbook = io.BytesIO()
    source.to_excel(workbook, index=False, engine="openpyxl")
    result = parse_uploaded_table(workbook.getvalue(), "sample.xlsx")
    assert result.iloc[0]["id"] == "00123"

    from openpyxl import Workbook
    formatted_workbook = Workbook()
    sheet = formatted_workbook.active
    sheet.append(["account_code", "amount"])
    sheet.append([123, 4])
    sheet["A2"].number_format = "00000"
    formatted = io.BytesIO()
    formatted_workbook.save(formatted)
    formatted_result = parse_uploaded_table(formatted.getvalue(), "formatted.xlsx")
    assert formatted_result.iloc[0]["account_code"] == "00123"


def test_reads_legacy_xls():
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("Customers")
    for column, value in enumerate(("id", "value")):
        sheet.write(0, column, value)
    sheet.write(1, 0, "00123")
    sheet.write(1, 1, 4)
    source = io.BytesIO()
    workbook.save(source)
    result = parse_uploaded_table(source.getvalue(), "sample.xls")
    assert result.iloc[0]["id"] == "00123"
    assert result.iloc[0]["value"] == 4


def test_rejects_bad_json_and_unsupported_formats():
    with pytest.raises(ValueError, match="malformed"):
        parse_uploaded_table(b"not-json", "sample.json")
    with pytest.raises(ValueError, match="Unsupported format"):
        parse_uploaded_table(b"x", "sample.xml")


def test_json_lines_requires_object_per_line():
    with pytest.raises(ValueError, match="must be an object"):
        parse_uploaded_table(b"[1,2]\n", "sample.jsonl")
