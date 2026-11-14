
from pathlib import Path
import pandas as pd


def _suggest_header_row(preview):
    """Suggest a header row when a workbook has a leading title/blank row."""

    if preview.empty:
        return 0

    populated = preview.notna().sum(axis=1)
    if len(populated) > 1 and populated.iloc[0] < 2:
        if populated.iloc[1] >= 2:
            return 1

    return 0


def _get_source_details(file_path):
    """Return the source name, extension, and reusable input object."""

    if hasattr(file_path, "name") and hasattr(file_path, "read"):
        filename = file_path.name
        file_path.seek(0)
        return filename, Path(filename).suffix.lower(), file_path

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return str(path), path.suffix.lower(), path


def inspect_workbook(file_path, preview_rows=5):
    """Inspect sheets and leading rows without selecting a sheet silently."""

    filename, extension, source = _get_source_details(file_path)

    if extension not in {".xlsx", ".xls"}:
        raise ValueError("Workbook inspection requires an Excel file.")

    sheets = []

    with pd.ExcelFile(source) as workbook:
        sheet_names = workbook.sheet_names

        for sheet_name in sheet_names:
            preview = pd.read_excel(
                source,
                sheet_name=sheet_name,
                header=None,
                nrows=preview_rows
            )
            sheets.append({
                "name": sheet_name,
                "rows_previewed": len(preview),
                "columns_previewed": len(preview.columns),
                "suggested_header_row": _suggest_header_row(preview),
                "preview": preview
            })

    return {
        "filename": filename,
        "sheet_names": sheet_names,
        "sheet_count": len(sheet_names),
        "sheets": sheets
    }


def load_dataset(file_path, sheet_name=None, header=0):
    """
    Load a CSV or Excel feedback dataset.

    Supports:
    1. Normal file paths (str or Path)
    2. Streamlit UploadedFile objects
    """

    filename, extension, source = _get_source_details(file_path)

    # Load dataset
    if extension == ".csv":
        if sheet_name is not None:
            raise ValueError("sheet_name is only supported for Excel files.")

        data = pd.read_csv(source, header=header)

    elif extension in [".xlsx", ".xls"]:
        with pd.ExcelFile(source) as workbook:
            sheet_names = workbook.sheet_names

        if sheet_name is None and len(sheet_names) > 1:
            available = ", ".join(sheet_names)
            raise ValueError(
                "The Excel workbook contains multiple sheets. "
                f"Choose one explicitly. Available sheets: {available}"
            )

        selected_sheet = (
            sheet_names[0]
            if sheet_name is None
            else sheet_name
        )

        if selected_sheet not in sheet_names:
            raise ValueError(
                f"Sheet '{selected_sheet}' was not found. "
                f"Available sheets: {', '.join(sheet_names)}"
            )

        if hasattr(source, "seek"):
            source.seek(0)

        header_preview = pd.read_excel(
            source,
            sheet_name=selected_sheet,
            header=None,
            nrows=8
        )
        suggested_header = _suggest_header_row(header_preview)

        if header == 0 and suggested_header != 0:
            raise ValueError(
                f"Sheet '{selected_sheet}' appears to have a leading "
                f"title or blank row. Load it with header={suggested_header} "
                "after inspecting the workbook."
            )

        if hasattr(source, "seek"):
            source.seek(0)

        data = pd.read_excel(
            source,
            sheet_name=selected_sheet,
            header=header
        )

    else:
        raise ValueError(
            "Only CSV and Excel files are supported."
        )

    # Validate dataset
    if data.empty:
        raise ValueError(
            f"The selected dataset is empty: {filename}"
        )

    return data
