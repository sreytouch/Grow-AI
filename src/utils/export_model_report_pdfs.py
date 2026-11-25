"""Export GrowAI crop-model CSV reports as readable PDF tables."""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A3, A4, landscape, portrait
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle


BRAND_GREEN = colors.HexColor("#24543A")
BRAND_SAGE = colors.HexColor("#E8F0E9")
INK = colors.HexColor("#17211B")
GRID = colors.HexColor("#B8C5BC")
MARGIN = 0.38 * inch
CELL_LIMIT = 90

REPORT_TITLES = {
    "crop_growing_context": "Crop Growing Context",
    "feature_importance_explained": "Feature Importance and Interpretation",
    "model_comparison": "Model Performance Comparison",
    "random_forest_classification_report": "Random Forest Classification Report",
    "random_forest_misclassifications": "Random Forest Misclassifications",
    "random_forest_test_predictions": "Random Forest Test Predictions",
    "sample_recommendations": "Sample Crop Recommendations",
}


def _layout(column_count: int):
    if column_count <= 3:
        return portrait(A4), 8.0, 9.5
    if column_count <= 7:
        return landscape(A4), 7.0, 8.5
    return landscape(A3), 5.7, 7.0


def _format_value(value, column: str) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        normalized_column = column.lower().replace(" ", "_")
        if normalized_column == "confidence":
            return f"{value:.1%}"
        if normalized_column == "importance_(%)":
            return f"{value:.2f}%"
        if "accuracy" in normalized_column or normalized_column in {
            "precision",
            "recall",
            "f1-score",
        }:
            return f"{value:.2%}"
        if normalized_column == "support" and value.is_integer():
            return f"{int(value)}"
        return f"{value:.3f}"
    return str(value)


def _safe_text(value: str) -> str:
    value = value[: CELL_LIMIT - 3] + "..." if len(value) > CELL_LIMIT else value
    return escape(value).replace("\n", "<br/>")


def _column_widths(df: pd.DataFrame, page_width: float) -> list[float]:
    sample = df.head(80)
    weights = []
    for column in df.columns:
        display_header = str(column).replace("_", " ").title()
        lengths = [len(display_header)]
        lengths.extend(len(_format_value(value, str(column))) for value in sample[column])
        cap = 44 if str(column).lower() == "interpretation" else 25
        weights.append(min(max(max(lengths), 7), cap))

    total = sum(weights) or 1
    usable_width = page_width - (2 * MARGIN)
    return [usable_width * weight / total for weight in weights]


def _footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.4)
    canvas.line(MARGIN, 0.34 * inch, document.pagesize[0] - MARGIN, 0.34 * inch)
    canvas.setFillColor(colors.HexColor("#53645A"))
    canvas.setFont("Helvetica", 7)
    canvas.drawString(MARGIN, 0.19 * inch, "GrowAI - Crop Recommendation Model")
    canvas.drawRightString(
        document.pagesize[0] - MARGIN,
        0.19 * inch,
        f"Page {document.page}",
    )
    canvas.restoreState()


def dataframe_to_pdf(df: pd.DataFrame, output_path: Path, title: str, source_name: str):
    page_size, body_size, leading = _layout(len(df.columns))
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "GrowAITitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=21,
        textColor=BRAND_GREEN,
        spaceAfter=5,
    )
    subtitle_style = ParagraphStyle(
        "GrowAISubtitle",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#53645A"),
    )
    body_style = ParagraphStyle(
        "GrowAITableBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=body_size,
        leading=leading,
        textColor=INK,
        spaceAfter=0,
    )
    header_style = ParagraphStyle(
        "GrowAITableHeader",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=page_size,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=0.42 * inch,
        bottomMargin=0.48 * inch,
        title=f"GrowAI - {title}",
        author="GrowAI",
        subject=f"PDF export of {source_name}",
    )

    headers = [str(column).replace("_", " ").title() for column in df.columns]
    table_data = [[Paragraph(_safe_text(header), header_style) for header in headers]]
    for row in df.itertuples(index=False, name=None):
        table_data.append(
            [
                Paragraph(_safe_text(_format_value(value, str(column))), body_style)
                for column, value in zip(df.columns, row)
            ]
        )

    table = LongTable(
        table_data,
        colWidths=_column_widths(df, page_size[0]),
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BRAND_GREEN),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BRAND_SAGE]),
                ("GRID", (0, 0), (-1, -1), 0.3, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )

    story = [
        Paragraph(f"GrowAI - {escape(title)}", title_style),
        Paragraph(
            f"Source: {escape(source_name)} | Rows: {len(df):,} | Columns: {len(df.columns)}",
            subtitle_style,
        ),
        Spacer(1, 0.18 * inch),
        table,
    ]
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)


def export_all_report_pdfs(reports_dir: Path) -> list[Path]:
    reports_dir = Path(reports_dir)
    output_paths = []
    for csv_path in sorted(reports_dir.glob("*.csv")):
        title = REPORT_TITLES.get(csv_path.stem, csv_path.stem.replace("_", " ").title())
        output_path = csv_path.with_suffix(".pdf")
        dataframe = pd.read_csv(csv_path)
        unnamed_columns = [column for column in dataframe.columns if str(column).startswith("Unnamed:")]
        if len(unnamed_columns) == 1:
            dataframe = dataframe.rename(columns={unnamed_columns[0]: "Class"})
        if {"Class", "support"}.issubset(dataframe.columns):
            dataframe["support"] = dataframe["support"].astype(object)
            dataframe.loc[dataframe["Class"] == "accuracy", "support"] = ""
        dataframe_to_pdf(dataframe, output_path, title, csv_path.name)
        output_paths.append(output_path)
    return output_paths


def main():
    if len(sys.argv) > 1:
        reports_dir = Path(sys.argv[1]).resolve()
    else:
        project_root = Path(__file__).resolve().parents[2]
        reports_dir = project_root / "crop_recommendation_results" / "reports"

    output_paths = export_all_report_pdfs(reports_dir)
    if not output_paths:
        raise FileNotFoundError(f"No CSV reports found in {reports_dir}")
    for output_path in output_paths:
        print(f"Created: {output_path}")


if __name__ == "__main__":
    main()
