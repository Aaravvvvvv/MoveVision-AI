from datetime import datetime
from pathlib import Path
import tempfile

import cv2
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from estimator import PACKING_TIPS, get_packing_tip, summarise_items


PDF_NAVY = colors.HexColor("#142850")
PDF_BLUE = colors.HexColor("#1e3a6e")
PDF_LIGHT = colors.HexColor("#f0f4f8")
PDF_ROW = colors.HexColor("#f5f7fa")
PDF_TEXT = colors.HexColor("#1f2937")


def _register_pdf_fonts():
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    regular = next((path for path in candidates if path.exists() and "Bold" not in path.name), None)
    bold = next((path for path in candidates if path.exists() and "Bold" in path.name), None)
    if regular:
        pdfmetrics.registerFont(TTFont("MoveVision", str(regular)))
        pdfmetrics.registerFont(TTFont("MoveVision-Bold", str(bold or regular)))
        return "MoveVision", "MoveVision-Bold"
    return "Helvetica", "Helvetica-Bold"


PDF_FONT, PDF_FONT_BOLD = _register_pdf_fonts()


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="MVTitle",
        parent=styles["Title"],
        fontName=PDF_FONT_BOLD,
        fontSize=22,
        textColor=colors.white,
        alignment=TA_CENTER,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="MVSubtitle",
        parent=styles["Normal"],
        fontName=PDF_FONT,
        fontSize=9,
        textColor=colors.white,
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name="MVHeading",
        parent=styles["Heading2"],
        fontName=PDF_FONT_BOLD,
        fontSize=12,
        textColor=PDF_BLUE,
        spaceBefore=12,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="MVNormal",
        parent=styles["Normal"],
        fontName=PDF_FONT,
        fontSize=8,
        leading=11,
        textColor=PDF_TEXT,
    ))
    return styles


def _table(data, widths=None, header=True):
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    table_style = [
        ("FONTNAME", (0, 0), (-1, -1), PDF_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, -1), PDF_TEXT),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        table_style.extend([
            ("BACKGROUND", (0, 0), (-1, 0), PDF_NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), PDF_FONT_BOLD),
        ])
    for row_idx in range(1 if header else 0, len(data)):
        if row_idx % 2 == 0:
            table_style.append(("BACKGROUND", (0, row_idx), (-1, row_idx), PDF_ROW))
    table.setStyle(TableStyle(table_style))
    return table


def _header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(PDF_NAVY)
    canvas.rect(0, height - 34, width, 34, fill=True, stroke=False)
    canvas.setFillColor(colors.white)
    canvas.setFont(PDF_FONT_BOLD, 10)
    canvas.drawString(36, height - 21, "MoveVision AI")
    canvas.setFont(PDF_FONT, 8)
    canvas.drawRightString(width - 36, height - 21, "Smart Household Inventory & Relocation Estimate")
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.setFont(PDF_FONT, 8)
    canvas.drawCentredString(width / 2, 24, f"Page {doc.page}")
    canvas.restoreState()


def _section(story, styles, title):
    story.append(Paragraph(title, styles["MVHeading"]))


def _merge_room_inventories(room_inventory):
    combined = {}
    for items in room_inventory.values():
        for item, count in items.items():
            combined[item] = combined.get(item, 0) + count
    return combined


def generate_report_pdf(room_inventory, summary, quote_data, room_frames=None, output_dir="outputs"):
    """Generate a Unicode-safe ReportLab PDF with page numbers and a summary table."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    path = output_path / "moving_estimate_preview.pdf"

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=54,
        bottomMargin=42,
    )
    styles = _styles()
    story = []

    title_box = Table(
        [[
            Paragraph("MoveVision AI", styles["MVTitle"]),
            Paragraph(f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}", styles["MVSubtitle"]),
        ]],
        colWidths=[3.7 * inch, 3.0 * inch],
    )
    title_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PDF_NAVY),
        ("BOX", (0, 0), (-1, -1), 0, PDF_NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(title_box)
    story.append(Spacer(1, 12))

    currency = quote_data.get("currency", "INR")
    story.append(_table([
        ["Route", f"{quote_data['origin']} -> {quote_data['destination']}", "Distance", f"{quote_data['distance_km']} km"],
        ["Truck", quote_data["truck"], "Volume", f"{quote_data['total_volume']} cu ft"],
        ["Weight", f"{quote_data['total_weight']} kg", "Fragile Items", str(quote_data["fragile_items"])],
        ["Estimate Range", f"{currency} {quote_data['lower_estimate']:,} - {currency} {quote_data['upper_estimate']:,}", "Best Estimate", f"{currency} {quote_data['midpoint']:,}"],
    ], widths=[1.25 * inch, 2.2 * inch, 1.25 * inch, 2.0 * inch], header=False))

    inv_header = ["Item", "Qty", "Category", "Fragility", "Volume (cu ft)", "Weight (kg)"]
    inv_widths = [1.35 * inch, 0.45 * inch, 1.25 * inch, 0.85 * inch, 1.1 * inch, 1.0 * inch]

    for room_name, items in room_inventory.items():
        if not items:
            continue
        room_summary = summarise_items(items)
        _section(story, styles, f"Room: {room_name}")
        rows = [inv_header]
        for item in room_summary["items"]:
            rows.append([
                item["Item"],
                item["Count"],
                item["Category"],
                item["Fragility"],
                item["Volume (cu ft)"],
                item["Weight (kg)"],
            ])
        rows.append(["Subtotal", "", "", "", room_summary["total_volume"], room_summary["total_weight"]])
        story.append(_table(rows, widths=inv_widths))

    _section(story, styles, "Combined Inventory")
    combined_rows = [inv_header]
    for item in summary["items"]:
        combined_rows.append([
            item["Item"],
            item["Count"],
            item["Category"],
            item["Fragility"],
            item["Volume (cu ft)"],
            item["Weight (kg)"],
        ])
    story.append(_table(combined_rows, widths=inv_widths))

    _section(story, styles, "Cost Estimate")
    story.append(_table([
        ["Cost Head", f"Amount ({currency})"],
        ["Base truck cost", f"{quote_data['base_cost']:,}"],
        ["Transport", f"{quote_data['transport_cost']:,}"],
        ["Packing material", f"{quote_data['packing_cost']:,}"],
        ["Loading & unloading", f"{quote_data['loading_cost']:,}"],
        ["Fragile surcharge", f"{quote_data['fragile_cost']:,}"],
        ["Floor surcharge", f"{quote_data['floor_cost']:,}"],
        ["Extra boxes", f"{quote_data['box_cost']:,}"],
    ], widths=[3.4 * inch, 1.6 * inch]))

    estimate_box = Table(
        [[
            Paragraph(f"Estimated Range: {currency} {quote_data['lower_estimate']:,} - {currency} {quote_data['upper_estimate']:,}", styles["MVNormal"]),
            Paragraph(f"Best Estimate: {currency} {quote_data['midpoint']:,}", styles["MVNormal"]),
        ]],
        colWidths=[3.5 * inch, 2.4 * inch],
    )
    estimate_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PDF_LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.5, PDF_BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(Spacer(1, 8))
    story.append(estimate_box)

    combined = _merge_room_inventories(room_inventory)
    fragile_items = [name for name in combined if name in PACKING_TIPS]
    if fragile_items:
        _section(story, styles, "Packing Tips")
        story.append(_table(
            [["Item", "Tip"]] + [[name.title(), get_packing_tip(name)] for name in fragile_items],
            widths=[1.5 * inch, 4.8 * inch],
        ))

    if room_frames:
        snapshots = []
        for room, frames in room_frames.items():
            for frame in frames[:1]:
                snapshots.append((room, frame))
        if snapshots:
            _section(story, styles, "Detection Snapshots")
            for room, frame in snapshots[:3]:
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                cv2.imwrite(tmp.name, frame)
                story.append(Paragraph(room, styles["MVNormal"]))
                story.append(Image(tmp.name, width=3.3 * inch, height=2.1 * inch))
                story.append(Spacer(1, 6))

    _section(story, styles, "Terms & Conditions")
    terms = [
        "1. This estimate is generated by AI-based object detection and may not reflect exact inventory.",
        "2. Final pricing is subject to physical survey and may vary based on actual items and conditions.",
        "3. Fragile items require special packing material which is included in the fragile surcharge.",
        "4. Floor charges apply only when lift is unavailable. Rate: INR 500 per floor above ground.",
        "5. Transit insurance is recommended for high-value items and is available at additional cost.",
        "6. The moving company reserves the right to adjust pricing based on access conditions at site.",
        "7. Cancellation within 24 hours of scheduled move may incur a cancellation fee of up to 20%.",
        "8. This report is valid for 30 days from the date of generation.",
    ]
    for term in terms:
        story.append(Paragraph(term, styles["MVNormal"]))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return path
