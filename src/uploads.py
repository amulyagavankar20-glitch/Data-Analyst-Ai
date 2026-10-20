"""Bounded readers for common tabular upload formats."""
import csv
import io
import json
import os
import re
import zipfile

import pandas as pd


MAX_ROWS = 500_000
MAX_COLUMNS = 200
SUPPORTED_FORMATS = ("CSV", "TSV", "XLS", "XLSX", "JSON", "JSONL", "NDJSON", "Parquet")


def parse_uploaded_table(raw: bytes, filename: str) -> pd.DataFrame:
    """Parse one bounded flat table; Excel imports its active sheet only."""
    suffix = os.path.splitext(filename.lower())[1]
    if suffix in {".csv", ".tsv"}:
        frame = _read_delimited(raw, "\t" if suffix == ".tsv" else ",")
    elif suffix in {".xlsx", ".xls"}:
        frame = _read_excel(raw, suffix)
    elif suffix in {".json", ".jsonl", ".ndjson"}:
        frame = _read_json(raw, suffix)
    elif suffix == ".parquet":
        frame = _read_parquet(raw)
    else:
        formats = ", ".join(SUPPORTED_FORMATS)
        raise ValueError(f"Unsupported format. Upload one of: {formats}.")

    if frame.empty or len(frame.columns) == 0:
        raise ValueError("The dataset has no usable rows or columns.")
    if len(frame) > MAX_ROWS or len(frame.columns) > MAX_COLUMNS:
        raise ValueError(f"Dataset exceeds the {MAX_ROWS:,}-row or {MAX_COLUMNS}-column limit.")
    if not frame.columns.is_unique:
        raise ValueError("Dataset contains duplicate column names.")
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed:")]
    if frame.empty or len(frame.columns) == 0:
        raise ValueError("The dataset has no usable columns.")
    return frame


def _read_delimited(raw: bytes, delimiter: str) -> pd.DataFrame:
    try:
        reader = csv.reader(io.StringIO(raw.decode("utf-8-sig")), delimiter=delimiter)
        header = next(reader, None)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError("The delimited file is malformed or is not UTF-8 encoded.") from exc
    if not header:
        raise ValueError("The delimited file is empty or has no header row.")
    if len(header) > MAX_COLUMNS:
        raise ValueError(f"Dataset exceeds the {MAX_COLUMNS}-column limit.")
    if len(header) != len(set(header)):
        raise ValueError("Delimited file contains duplicate column names.")
    rows = 0
    zero_padded_columns = set()
    try:
        for row in reader:
            rows += 1
            if rows > MAX_ROWS:
                raise ValueError(f"Dataset exceeds the {MAX_ROWS:,}-row limit.")
            for index, value in enumerate(row[:len(header)]):
                if re.fullmatch(r"[+-]?0\d+", value.strip()):
                    zero_padded_columns.add(header[index])
    except csv.Error as exc:
        raise ValueError("The delimited file is malformed.") from exc
    return pd.read_csv(io.BytesIO(raw), sep=delimiter, nrows=MAX_ROWS + 1,
                       dtype={column: "string" for column in zero_padded_columns})


def _read_excel(raw: bytes, suffix: str) -> pd.DataFrame:
    zero_widths = {}
    if suffix == ".xlsx":
        try:
            archive = zipfile.ZipFile(io.BytesIO(raw))
        except zipfile.BadZipFile as exc:
            raise ValueError("The Excel workbook is malformed.") from exc
        with archive:
            members = archive.infolist()
            if (len(members) > 10_000 or sum(item.file_size for item in members) > 128 * 1024 * 1024
                    or any(item.file_size > 64 * 1024 * 1024
                           or (item.compress_size == 0 and item.file_size > 0)
                           or (item.compress_size and item.file_size / item.compress_size > 200)
                           for item in members)):
                raise ValueError("Workbook archive exceeds decompression limits.")
        import openpyxl
        workbook = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
        try:
            sheet = workbook.active
            if sheet.max_column > MAX_COLUMNS or sheet.max_row > MAX_ROWS + 1:
                raise ValueError(f"Dataset exceeds the {MAX_ROWS:,}-row or {MAX_COLUMNS}-column limit.")
            header = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True), None) or []
            cleaned = [item for item in header if item is not None]
            if len(cleaned) != len(set(cleaned)):
                raise ValueError("Excel sheet contains duplicate column names.")
            for row in sheet.iter_rows(min_row=2, max_row=min(sheet.max_row, 101)):
                for cell in row:
                    if (isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool)
                            and re.fullmatch(r"0{2,}", cell.number_format or "")):
                        zero_widths[cell.column - 1] = max(
                            zero_widths.get(cell.column - 1, 0), len(cell.number_format)
                        )
        finally:
            workbook.close()
    else:
        import xlrd
        try:
            workbook = xlrd.open_workbook(file_contents=raw, on_demand=True)
            try:
                sheet = workbook.sheet_by_index(0)
                if sheet.ncols > MAX_COLUMNS or sheet.nrows > MAX_ROWS + 1:
                    raise ValueError(f"Dataset exceeds the {MAX_ROWS:,}-row or {MAX_COLUMNS}-column limit.")
                header = sheet.row_values(0) if sheet.nrows else []
                if len(header) != len(set(header)):
                    raise ValueError("Excel sheet contains duplicate column names.")
            finally:
                workbook.release_resources()
        except xlrd.XLRDError as exc:
            raise ValueError("The Excel workbook is malformed or unreadable.") from exc
    engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
    # Read Excel cells without pandas' eager numeric inference so text IDs like
    # "00123" remain intact, then infer only unambiguous numeric columns.
    frame = pd.read_excel(io.BytesIO(raw), nrows=MAX_ROWS + 1,
                          engine=engine, dtype=object)
    for column in frame.columns:
        values = frame[column]
        if values.map(lambda value: isinstance(value, bool)).all():
            continue
        if values.map(lambda value: isinstance(value, str)
                      and re.fullmatch(r"[+-]?0\d+", value.strip()) is not None).any():
            continue
        numeric = pd.to_numeric(values, errors="coerce")
        if values.notna().equals(numeric.notna()):
            frame[column] = numeric
    for column_index, width in zero_widths.items():
        if column_index >= frame.shape[1]:
            continue
        column = frame.columns[column_index]
        def preserve_formatted_number(value):
            if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value).is_integer():
                return str(int(value)).zfill(width)
            return value
        frame[column] = frame[column].map(preserve_formatted_number)
    return frame


def _read_json(raw: bytes, suffix: str) -> pd.DataFrame:
    try:
        if suffix in {".jsonl", ".ndjson"}:
            records = []
            for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"JSON Lines entry {line_number} must be an object.")
                records.append(record)
                if len(records) > MAX_ROWS:
                    raise ValueError(f"Dataset exceeds the {MAX_ROWS:,}-row limit.")
            return pd.DataFrame.from_records(records)

        document = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("The JSON file is malformed or is not UTF-8 encoded.") from exc
    if isinstance(document, list) and all(isinstance(row, dict) for row in document):
        if len(document) > MAX_ROWS:
            raise ValueError(f"Dataset exceeds the {MAX_ROWS:,}-row limit.")
        return pd.DataFrame.from_records(document)
    if isinstance(document, dict) and document and all(isinstance(values, list) for values in document.values()):
        lengths = {len(values) for values in document.values()}
        if len(lengths) != 1:
            raise ValueError("JSON columns must all have the same number of values.")
        if lengths and next(iter(lengths)) > MAX_ROWS:
            raise ValueError(f"Dataset exceeds the {MAX_ROWS:,}-row limit.")
        return pd.DataFrame(document)
    raise ValueError("JSON must contain an array of row objects or an object of equal-length column arrays.")


def _read_parquet(raw: bytes) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.parquet as pq
    try:
        metadata = pq.ParquetFile(io.BytesIO(raw)).metadata
        if metadata.num_rows > MAX_ROWS or metadata.num_columns > MAX_COLUMNS:
            raise ValueError(f"Dataset exceeds the {MAX_ROWS:,}-row or {MAX_COLUMNS}-column limit.")
        return pd.read_parquet(io.BytesIO(raw), engine="pyarrow")
    except ValueError as exc:
        if "limit" in str(exc):
            raise
        raise ValueError("The Parquet file is malformed or unreadable.") from exc
    except (OSError, pa.ArrowException) as exc:
        raise ValueError("The Parquet file is malformed or unreadable.") from exc
