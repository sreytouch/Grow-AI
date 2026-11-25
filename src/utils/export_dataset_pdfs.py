"""
Export GrowAI datasets to PDF tables and manifests.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import legal, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.utils.config import DATA_PROCESSED, DATA_RAW, PDF_OUTPUT_DIR

PAGE_SIZE = landscape(legal)
PAGE_WIDTH = PAGE_SIZE[0] - 0.7 * inch
ROWS_PER_PAGE = 24
CELL_LIMIT = 80

styles = getSampleStyleSheet()
TITLE_STYLE = styles["Title"]
BODY_STYLE = ParagraphStyle(
    "DatasetBody",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=7,
    leading=8.5,
    spaceAfter=0,
)
HEADER_STYLE = ParagraphStyle(
    "DatasetHeader",
    parent=BODY_STYLE,
    fontName="Helvetica-Bold",
    fontSize=7.5,
    leading=9,
    textColor=colors.white,
)


def _cell_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return escape(text[: CELL_LIMIT - 3] + "...") if len(text) > CELL_LIMIT else escape(text)


def _paragraph(value, style):
    return Paragraph(_cell_text(value).replace("\n", "<br/>"), style)


def _column_widths(headers: list[str], rows: list[list[str]]) -> list[float]:
    sample_rows = rows[: min(len(rows), 12)]
    scores = []
    for idx, header in enumerate(headers):
        lengths = [len(header)] + [len(str(row[idx])) for row in sample_rows]
        scores.append(min(max(max(lengths), 10), 28))

    total = sum(scores) or 1
    return [PAGE_WIDTH * score / total for score in scores]


def _build_table(headers: list[str], rows: list[list[str]]) -> Table:
    table_rows = [[_paragraph(header, HEADER_STYLE) for header in headers]]
    table_rows.extend(
        [[_paragraph(value, BODY_STYLE) for value in row] for row in rows]
    )
    table = Table(table_rows, colWidths=_column_widths(headers, rows), repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E5B3B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def dataframe_to_pdf(df: pd.DataFrame, output_path: Path, title: str, subtitle: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=PAGE_SIZE,
        leftMargin=0.35 * inch,
        rightMargin=0.35 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
    )
    story = [Paragraph(title, TITLE_STYLE), Spacer(1, 0.15 * inch), Paragraph(subtitle, styles["BodyText"]), Spacer(1, 0.2 * inch)]

    printable = df.copy()
    printable.columns = [str(col) for col in printable.columns]
    rows = printable.astype(object).where(pd.notna(printable), "").astype(str).values.tolist()
    headers = printable.columns.tolist()

    for start in range(0, len(rows), ROWS_PER_PAGE):
        chunk = rows[start:start + ROWS_PER_PAGE]
        story.append(_build_table(headers, chunk))
        if start + ROWS_PER_PAGE < len(rows):
            story.extend([Spacer(1, 0.12 * inch), PageBreak()])

    document.build(story)


def export_csv_pdf(input_path: Path, output_name: str, title: str):
    df = pd.read_csv(input_path)
    dataframe_to_pdf(
        df,
        PDF_OUTPUT_DIR / output_name,
        title,
        f"Source: {input_path.name} | Rows: {len(df):,} | Columns: {len(df.columns)}",
    )


def export_json_pdf(input_path: Path, output_name: str, title: str):
    records = json.loads(input_path.read_text(encoding="utf-8"))
    df = pd.DataFrame(records)
    dataframe_to_pdf(
        df,
        PDF_OUTPUT_DIR / output_name,
        title,
        f"Source: {input_path.name} | Entries: {len(df):,} | Columns: {len(df.columns)}",
    )


def export_plantvillage_manifest(output_name: str = "processed_plantvillage_manifest.pdf"):
    root = DATA_PROCESSED / "plantvillage_split"
    records = []
    for split_dir in sorted(root.iterdir()):
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() not in {".jpg", ".png"}:
                    continue
                records.append(
                    {
                        "split": split_dir.name,
                        "class_name": class_dir.name,
                        "file_name": image_path.name,
                        "relative_path": str(image_path.relative_to(root)),
                    }
                )

    df = pd.DataFrame(records)
    dataframe_to_pdf(
        df,
        PDF_OUTPUT_DIR / output_name,
        "GrowAI PlantVillage Manifest",
        f"Source: {root.name} | Images listed: {len(df):,}",
    )


def run_exports():
    export_csv_pdf(DATA_RAW / "Crop_recommendation.csv", "raw_crop_recommendation.pdf", "GrowAI Raw Crop Recommendation Dataset")
    export_csv_pdf(DATA_RAW / "indoor_plants.csv", "raw_indoor_plants.pdf", "GrowAI Raw Indoor Plants Dataset")
    export_csv_pdf(DATA_PROCESSED / "usda_nass_clean.csv", "processed_usda_nass_clean.pdf", "GrowAI USDA NASS Clean Dataset")
    export_csv_pdf(DATA_PROCESSED / "kaggle_agri_clean.csv", "processed_kaggle_agri_clean.pdf", "GrowAI Kaggle Agriculture Clean Dataset")
    export_csv_pdf(DATA_PROCESSED / "weather_33.77_-118.19.csv", "processed_weather_33.77_-118.19.pdf", "GrowAI Weather Dataset (Long Beach, CA)")
    export_json_pdf(DATA_PROCESSED / "plant_knowledge_base.json", "processed_plant_knowledge_base.pdf", "GrowAI Plant Knowledge Base")
    export_plantvillage_manifest()
    print(f"Saved PDF exports to {PDF_OUTPUT_DIR}")


if __name__ == "__main__":
    run_exports()
