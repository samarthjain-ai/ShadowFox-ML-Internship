
from pathlib import Path
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


def export_results(
    results,
    output_path="outputs/reviewmind_analysis.csv"
):
    """Export the analyzed feedback DataFrame to CSV."""

    if not isinstance(results, dict):
        raise TypeError("Results must be a dictionary.")

    if "results_dataframe" not in results:
        raise ValueError(
            "No results DataFrame found to export."
        )

    dataframe = results["results_dataframe"]

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "The results_dataframe must be a pandas DataFrame."
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    return str(output_path.resolve())


def build_report_pdf(data, results=None, profile=None):
    """Create a dependency-light PDF summary from verified dataframe values."""

    results = results or {}
    profile = profile or {}
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        figure = plt.figure(figsize=(8.27, 11.69))
        axis = figure.add_axes([0.1, 0.1, 0.8, 0.82])
        axis.axis("off")
        lines = [
            "ReviewMind AI — Analysis Report",
            "",
            f"Rows: {len(data):,}",
            f"Columns: {len(data.columns):,}",
            f"Missing values: {profile.get('missing_value_total', int(data.isna().sum().sum())):,}",
            f"Analyzed rows: {results.get('row_count_analyzed', 'Not run')}",
            "",
            "Columns:",
            ", ".join(map(str, data.columns)),
            "",
            "Completed analyses:",
            ", ".join(
                name for name, value in results.get("analysis_summary", {}).items()
                if str(value).startswith("completed")
            ) or "No analysis has been run.",
            "",
            "Limitations:",
            "Predictions are model outputs and should be reviewed alongside the original feedback.",
        ]
        axis.text(0, 1, "\n".join(lines), va="top", fontsize=11, color="#1d2136")
        pdf.savefig(figure)
        plt.close(figure)
    return buffer.getvalue()
