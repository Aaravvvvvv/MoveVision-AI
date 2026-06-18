from datetime import datetime
from pathlib import Path
from time import time
import re
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF

from detector import HOUSEHOLD_ITEMS, MODEL_PATH, process_image, process_video
from estimator import summarise_items
import estimator as _estimator
PACKING_TIPS = getattr(_estimator, "PACKING_TIPS", {})
get_packing_tip = getattr(_estimator, "get_packing_tip", lambda n: "Wrap securely and label.")
from quote import calculate_quote, get_distance


# ── Constants ────────────────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
CITY_OPTIONS = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Hyderabad", "Pune", "Kolkata"]
ROOM_OPTIONS = [
    "Living Room", "Bedroom", "Kitchen", "Bathroom",
    "Dining Room", "Office", "Balcony", "Storage", "Other",
]


# ── Helpers ──────────────────────────────────────────────────────────
def safe_filename(name):
    stem = Path(name).stem
    suffix = Path(name).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "upload"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{timestamp}{suffix}"


def save_uploaded_file(uploaded_file):
    path = UPLOAD_DIR / safe_filename(uploaded_file.name)
    path.write_bytes(uploaded_file.getvalue())
    return path


def frame_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def detected_items_to_frame(detected_items, confidence_summary=None):
    confidence_summary = confidence_summary or {}
    rows = [
        {
            "Item": item,
            "Count": count,
            "Confidence": (
                f"{confidence_summary[item]['avg']:.0%}"
                if item in confidence_summary else "Manual"
            ),
            "Review Status": (
                confidence_summary[item]["level"]
                if item in confidence_summary else "Manual"
            ),
        }
        for item, count in sorted(detected_items.items())
    ]
    return pd.DataFrame(rows, columns=["Item", "Count", "Confidence", "Review Status"])


def confidence_status_counts(confidence_by_room):
    counts = {"High": 0, "Review": 0, "Low": 0, "Manual": 0}
    for room_data in confidence_by_room.values():
        for stats in room_data.values():
            level = stats.get("level", "Manual")
            counts[level] = counts.get(level, 0) + 1
    return counts


def confidence_review_rows(confidence_by_room):
    rows = []
    for room_name, room_data in confidence_by_room.items():
        for item_name, stats in room_data.items():
            level = stats.get("level", "Manual")
            if level == "High":
                continue
            rows.append({
                "Room": room_name,
                "Item": item_name,
                "Confidence": f"{stats.get('avg', 0):.0%}",
                "Review Status": level,
            })
    return rows


def editor_frame_to_items(frame):
    items = {}
    for _, row in frame.fillna("").iterrows():
        item = str(row.get("Item", "")).strip().lower()
        if not item or item in {"none", "nan", "null"}:
            continue
        try:
            count = int(row.get("Count", 0))
        except (TypeError, ValueError):
            count = 0
        if count > 0:
            items[item] = items.get(item, 0) + count
    return items


def merge_room_inventories(room_inventory):
    """Flatten room-keyed inventory into a single {item: count} dict."""
    combined = {}
    for items in room_inventory.values():
        for item, count in items.items():
            combined[item] = combined.get(item, 0) + count
    return combined


# ── Premium PDF Report ───────────────────────────────────────────────
# Color palette
_NAVY      = (20, 40, 80)
_DARK_BLUE = (30, 58, 110)
_LIGHT_BG  = (240, 244, 248)
_WHITE     = (255, 255, 255)
_GREY_ROW  = (245, 247, 250)
_TEXT      = (30, 30, 30)
_ACCENT    = (50, 120, 200)


def _pdf_safe_text(value):
    """Convert UI/model text to characters supported by FPDF's core fonts."""
    if value is None:
        return ""
    text = str(value)
    replacements = {
        "₹": "INR ",
        "→": "->",
        "–": "-",
        "—": "-",
        "’": "'",
        "“": '"',
        "”": '"',
        "•": "-",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text.encode("latin-1", errors="replace").decode("latin-1")


class SafeFPDF(FPDF):
    """FPDF wrapper that prevents Unicode crashes when using core fonts."""

    def cell(self, w=None, h=None, text="", *args, **kwargs):
        return super().cell(w, h, _pdf_safe_text(text), *args, **kwargs)


def _pdf_section_header(pdf, title):
    """Draw a styled section header with colored background."""
    pdf.set_fill_color(*_DARK_BLUE)
    pdf.set_text_color(*_WHITE)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, f"  {title}", ln=True, fill=True)
    pdf.set_text_color(*_TEXT)
    pdf.ln(2)


def _pdf_table_header(pdf, columns):
    """Draw a table header row."""
    pdf.set_fill_color(*_NAVY)
    pdf.set_text_color(*_WHITE)
    pdf.set_font("Arial", "B", 8)
    for col_name, width in columns:
        pdf.cell(width, 7, col_name, border=0, fill=True)
    pdf.ln()
    pdf.set_text_color(*_TEXT)


def _pdf_table_row(pdf, values, columns, row_index):
    """Draw a table data row with alternating background."""
    if row_index % 2 == 0:
        pdf.set_fill_color(*_GREY_ROW)
    else:
        pdf.set_fill_color(*_WHITE)
    pdf.set_font("Arial", "", 8)
    for val, (_, width) in zip(values, columns):
        pdf.cell(width, 6, str(val)[:28], border=0, fill=True)
    pdf.ln()


def generate_report_pdf(room_inventory, summary, quote_data, room_frames=None):
    """Generate a premium styled PDF report with room-wise breakdown."""
    path = OUTPUT_DIR / "moving_estimate_preview.pdf"

    pdf = SafeFPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Header banner ────────────────────────────────────────────
    pdf.set_fill_color(*_NAVY)
    pdf.rect(0, 0, 210, 38, "F")
    pdf.set_text_color(*_WHITE)
    pdf.set_font("Arial", "B", 22)
    pdf.set_y(8)
    pdf.cell(0, 10, "  MoveVision AI", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, "  Smart Household Inventory & Relocation Estimate", ln=True)
    pdf.set_font("Arial", "", 9)
    pdf.cell(0, 6, f"  Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}", ln=True)
    pdf.set_text_color(*_TEXT)
    pdf.ln(6)

    # ── Move summary ─────────────────────────────────────────────
    _pdf_section_header(pdf, "Move Summary")
    pdf.set_font("Arial", "", 10)
    info_rows = [
        ("Route", f"{quote_data['origin']}  →  {quote_data['destination']}"),
        ("Distance", f"{quote_data['distance_km']} km"),
        ("Recommended Truck", quote_data["truck"]),
        ("Total Volume", f"{quote_data['total_volume']} cu ft"),
        ("Total Weight", f"{quote_data['total_weight']} kg"),
        ("Fragile Items", str(quote_data["fragile_items"])),
    ]
    for label, value in info_rows:
        pdf.set_font("Arial", "B", 9)
        pdf.cell(45, 6, label)
        pdf.set_font("Arial", "", 9)
        pdf.cell(0, 6, value, ln=True)
    pdf.ln(4)

    # ── Room-by-room inventory ───────────────────────────────────
    inv_cols = [("Item", 40), ("Qty", 14), ("Category", 30),
                ("Fragility", 22), ("Volume (cu ft)", 28), ("Weight (kg)", 28)]

    for room_name, items in room_inventory.items():
        if not items:
            continue
        room_summary = summarise_items(items)

        _pdf_section_header(pdf, f"Room: {room_name}")
        _pdf_table_header(pdf, inv_cols)

        for idx, item in enumerate(room_summary["items"]):
            _pdf_table_row(pdf, [
                item["Item"], item["Count"], item["Category"],
                item["Fragility"], item["Volume (cu ft)"], item["Weight (kg)"],
            ], inv_cols, idx)

        pdf.set_font("Arial", "I", 8)
        pdf.set_text_color(100, 100, 100)
        rv = room_summary["total_volume"]
        rw = room_summary["total_weight"]
        pdf.cell(0, 6, f"  Subtotal:  {rv} cu ft  /  {rw} kg", ln=True)
        pdf.set_text_color(*_TEXT)
        pdf.ln(3)

    # ── Combined inventory totals ────────────────────────────────
    _pdf_section_header(pdf, "Combined Inventory")
    _pdf_table_header(pdf, inv_cols)
    for idx, item in enumerate(summary["items"]):
        _pdf_table_row(pdf, [
            item["Item"], item["Count"], item["Category"],
            item["Fragility"], item["Volume (cu ft)"], item["Weight (kg)"],
        ], inv_cols, idx)
    pdf.ln(4)

    # ── Cost estimate ────────────────────────────────────────────
    _pdf_section_header(pdf, "Cost Estimate")
    cost_cols = [("Cost Head", 80), ("Amount (INR)", 40)]
    _pdf_table_header(pdf, cost_cols)
    cost_rows = [
        ("Base truck cost", f"{quote_data['base_cost']:,}"),
        ("Transport", f"{quote_data['transport_cost']:,}"),
        ("Packing material", f"{quote_data['packing_cost']:,}"),
        ("Loading & unloading", f"{quote_data['loading_cost']:,}"),
        ("Fragile surcharge", f"{quote_data['fragile_cost']:,}"),
        ("Floor surcharge", f"{quote_data['floor_cost']:,}"),
        ("Extra boxes", f"{quote_data['box_cost']:,}"),
    ]
    for idx, (label, amount) in enumerate(cost_rows):
        _pdf_table_row(pdf, [label, amount], cost_cols, idx)
    pdf.ln(3)

    # Estimate range box
    pdf.set_fill_color(*_LIGHT_BG)
    pdf.set_draw_color(*_ACCENT)
    pdf.rect(pdf.get_x(), pdf.get_y(), 190, 22, "DF")
    y_box = pdf.get_y()
    pdf.set_xy(pdf.get_x() + 5, y_box + 2)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(*_NAVY)
    pdf.cell(0, 8, f"Estimated Range:  INR {quote_data['lower_estimate']:,}  -  INR {quote_data['upper_estimate']:,}", ln=True)
    pdf.set_x(pdf.get_x() + 5)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 7, f"Best Estimate:  INR {quote_data['midpoint']:,}", ln=True)
    pdf.set_text_color(*_TEXT)
    pdf.ln(6)

    # ── Packing tips ─────────────────────────────────────────────
    combined = merge_room_inventories(room_inventory)
    fragile_items = [
        name for name, _ in combined.items()
        if name in PACKING_TIPS
    ]
    if fragile_items:
        _pdf_section_header(pdf, "Packing Tips")
        pdf.set_font("Arial", "", 8)
        for idx, item_name in enumerate(fragile_items):
            tip = get_packing_tip(item_name)
            if idx % 2 == 0:
                pdf.set_fill_color(*_GREY_ROW)
            else:
                pdf.set_fill_color(*_WHITE)
            pdf.set_font("Arial", "B", 8)
            pdf.cell(35, 5, f"  {item_name.title()}", fill=True)
            pdf.set_font("Arial", "", 8)
            pdf.cell(0, 5, tip[:90], fill=True, ln=True)
        pdf.ln(4)

    # ── Annotated frame thumbnails ───────────────────────────────
    if room_frames:
        all_frames = []
        for room, frames in room_frames.items():
            for f in frames[:1]:  # max 1 per room
                all_frames.append((room, f))
        if all_frames:
            _pdf_section_header(pdf, "Detection Snapshots")
            for room, frame in all_frames[:3]:
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                cv2.imwrite(tmp.name, frame)
                try:
                    pdf.set_font("Arial", "I", 8)
                    pdf.cell(0, 5, f"  {room}", ln=True)
                    pdf.image(tmp.name, x=10, w=90)
                    pdf.ln(3)
                except Exception:
                    pass
            pdf.ln(2)

    # ── Terms & conditions ───────────────────────────────────────
    _pdf_section_header(pdf, "Terms & Conditions")
    pdf.set_font("Arial", "", 7)
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
    for line in terms:
        pdf.cell(0, 4, line, ln=True)
    pdf.ln(3)

    # Footer
    pdf.set_font("Arial", "I", 7)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 5, "Powered by MoveVision AI  |  movevision.ai  |  Report generated automatically", ln=True)

    pdf.output(str(path))
    return path


# ── Streamlit Page Config ────────────────────────────────────────────
st.set_page_config(page_title="MoveVision AI", page_icon="📦", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.2rem;}
    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 14px;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] label p {
        color: #d1d5db !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] div {
        color: #f9fafb !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricDelta"],
    div[data-testid="stMetricDelta"] div {
        color: #86efac !important;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📦 MoveVision AI")
st.caption("Smart Household Inventory & Relocation Estimator")


# ── Session State Init ───────────────────────────────────────────────
if "room_inventory" not in st.session_state:
    st.session_state.room_inventory = {}
if "room_frames" not in st.session_state:
    st.session_state.room_frames = {}
if "room_confidence" not in st.session_state:
    st.session_state.room_confidence = {}
if "scan_done" not in st.session_state:
    st.session_state.scan_done = False


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Scan")
    uploaded_files = st.file_uploader(
        "Upload room photos or videos",
        type=sorted([ext.lstrip(".") for ext in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS]),
        accept_multiple_files=True,
    )

    # Room assignment for each uploaded file
    file_rooms = {}
    if uploaded_files:
        st.markdown("**Assign rooms:**")
        for idx, f in enumerate(uploaded_files):
            room = st.selectbox(
                f.name,
                ROOM_OPTIONS,
                index=0,
                key=f"room_{idx}_{f.name}",
            )
            file_rooms[f.name] = room

    st.divider()
    st.markdown("**Detection settings**")
    confidence = st.slider("Confidence", 0.05, 0.90, 0.25, 0.05)
    imgsz = st.select_slider("Inference size", options=[640, 960, 1280], value=960)
    frame_interval = st.number_input("Video frame interval", min_value=1, max_value=120, value=15, step=1)
    st.caption(f"Model: {MODEL_PATH.name}")

    st.divider()
    st.header("🚚 Move Details")
    origin = st.selectbox("Origin", CITY_OPTIONS, index=0)
    destination = st.selectbox("Destination", CITY_OPTIONS, index=1)
    default_distance = get_distance(origin, destination)
    distance_km = st.number_input("Distance (km)", min_value=1, max_value=5000, value=int(default_distance), step=10)
    floor_number = st.number_input("Floor", min_value=0, max_value=50, value=0, step=1)
    has_lift = st.checkbox("Lift available", value=True)
    extra_boxes = st.number_input("Extra packed boxes", min_value=0, max_value=300, value=0, step=1)
    box_type = st.selectbox("Box type", ["light", "heavy"], index=0)

    st.divider()
    run_scan = st.button("🚀 Run Scan", type="primary", width="stretch")
    clear_scan = st.button("🗑️ Clear Results", width="stretch")


# ── Clear handler ────────────────────────────────────────────────────
if clear_scan:
    st.session_state.room_inventory = {}
    st.session_state.room_frames = {}
    st.session_state.room_confidence = {}
    st.session_state.scan_done = False
    st.rerun()


# ── Scan handler ─────────────────────────────────────────────────────
if run_scan:
    if not uploaded_files:
        st.warning("Upload at least one image or video first.")
    else:
        new_inventory = {}
        new_frames = {}
        new_confidence = {}
        total_files = len(uploaded_files)

        for file_idx, uploaded_file in enumerate(uploaded_files):
            room = file_rooms.get(uploaded_file.name, "Other")
            upload_path = save_uploaded_file(uploaded_file)
            suffix = upload_path.suffix.lower()

            st.markdown(f"**Scanning {file_idx + 1}/{total_files}:** `{uploaded_file.name}` → _{room}_")

            if suffix in IMAGE_EXTENSIONS:
                with st.spinner(f"Detecting items in {uploaded_file.name}..."):
                    detected_items, annotated, confidence_summary = process_image(
                        str(upload_path), confidence=confidence, imgsz=imgsz,
                        return_details=True,
                    )
                    frames = [annotated]

            elif suffix in VIDEO_EXTENSIONS:
                progress_bar = st.progress(0, text=f"Processing video: {uploaded_file.name}")
                status_text = st.empty()
                start_time = time()

                def _progress(frame_num, total):
                    pct = min(frame_num / max(total, 1), 1.0)
                    elapsed = time() - start_time
                    if pct > 0:
                        eta = elapsed / pct * (1 - pct)
                        eta_str = f"{int(eta)}s remaining"
                    else:
                        eta_str = "calculating..."
                    progress_bar.progress(pct, text=f"Frame {frame_num}/{total}")
                    status_text.caption(f"⏱️ {elapsed:.0f}s elapsed  |  {eta_str}")

                detected_items, frames, confidence_summary = process_video(
                    str(upload_path),
                    frame_interval=int(frame_interval),
                    confidence=confidence,
                    imgsz=imgsz,
                    progress_callback=_progress,
                    return_details=True,
                )
                progress_bar.progress(1.0, text="✅ Complete")
                status_text.empty()
            else:
                st.error(f"Unsupported file: {uploaded_file.name}")
                continue

            # Merge into room inventory
            if room not in new_inventory:
                new_inventory[room] = {}
            for item, count in detected_items.items():
                new_inventory[room][item] = new_inventory[room].get(item, 0) + count

            if room not in new_confidence:
                new_confidence[room] = {}
            for item, stats in confidence_summary.items():
                existing = new_confidence[room].get(item)
                if existing is None or stats["avg"] > existing["avg"]:
                    new_confidence[room][item] = stats

            # Store frames
            if room not in new_frames:
                new_frames[room] = []
            new_frames[room].extend(frames[:3])

        st.session_state.room_inventory = new_inventory
        st.session_state.room_frames = new_frames
        st.session_state.room_confidence = new_confidence
        st.session_state.scan_done = True
        st.rerun()


# ── Status bar ───────────────────────────────────────────────────────
if st.session_state.scan_done:
    total_rooms = len(st.session_state.room_inventory)
    total_items = sum(
        sum(items.values())
        for items in st.session_state.room_inventory.values()
    )
    st.success(f"✅ Scan complete — {total_items} items detected across {total_rooms} room(s)")
    confidence_counts = confidence_status_counts(st.session_state.room_confidence)
    c1, c2, c3 = st.columns(3)
    c1.metric("High-confidence item types", confidence_counts.get("High", 0))
    c2.metric("Needs review", confidence_counts.get("Review", 0))
    c3.metric("Low-confidence item types", confidence_counts.get("Low", 0))


# ── Main area — Tabs ─────────────────────────────────────────────────
tab_inventory, tab_estimate, tab_frames = st.tabs(["📋 Inventory", "💰 Estimate", "🖼️ Frames"])


# ── Tab 1: Inventory ─────────────────────────────────────────────────
with tab_inventory:
    room_inv = st.session_state.room_inventory

    if not room_inv:
        # Show a single default editor for manual entry
        st.subheader("Inventory")
        st.info("Upload and scan room photos, or add items manually below.")
        empty_df = pd.DataFrame(columns=["Item", "Count"])
        edited = st.data_editor(
            empty_df,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "Item": st.column_config.SelectboxColumn("Item", options=sorted(HOUSEHOLD_ITEMS), required=True),
                "Count": st.column_config.NumberColumn("Count", min_value=1, max_value=100, step=1, required=True),
            },
            key="manual_editor",
        )
        manual_items = editor_frame_to_items(edited)
        if manual_items:
            st.session_state.room_inventory = {"Manual Entry": manual_items}
    else:
        st.subheader("Room-by-Room Inventory")
        st.caption("Edit items and counts per room. Rows marked Review or Low should be checked before using the quote.")

        review_rows = confidence_review_rows(st.session_state.room_confidence)
        if review_rows:
            st.warning("Some detections need manual review. Confirm the item name and count in the inventory table below.")
            st.dataframe(pd.DataFrame(review_rows), width="stretch", hide_index=True)

        edited_room_inventory = {}

        for room_name in sorted(room_inv.keys()):
            items = room_inv[room_name]
            item_count = sum(items.values())
            category_count = len(items)

            with st.expander(f"🏠 {room_name}  —  {item_count} items, {category_count} types", expanded=True):
                df = detected_items_to_frame(
                    items,
                    st.session_state.room_confidence.get(room_name, {}),
                )
                edited_df = st.data_editor(
                    df,
                    num_rows="dynamic",
                    width="stretch",
                    hide_index=True,
                    disabled=["Confidence", "Review Status"],
                    column_config={
                        "Item": st.column_config.SelectboxColumn(
                            "Item", options=sorted(HOUSEHOLD_ITEMS), required=True,
                        ),
                        "Count": st.column_config.NumberColumn(
                            "Count", min_value=1, max_value=100, step=1, required=True,
                        ),
                        "Confidence": st.column_config.TextColumn("Confidence"),
                        "Review Status": st.column_config.TextColumn("Review Status"),
                    },
                    key=f"editor_{room_name}",
                )
                edited_items = editor_frame_to_items(edited_df)
                if edited_items:
                    edited_room_inventory[room_name] = edited_items

        # Combined summary
        combined = merge_room_inventories(edited_room_inventory)
        if combined:
            st.divider()
            st.subheader("Combined Summary")
            summary = summarise_items(combined)
            summary_df = pd.DataFrame(summary["items"])
            total_count = sum(combined.values())
            st.caption(f"{total_count} total items across {len(combined)} categories")
            st.dataframe(summary_df, width="stretch", hide_index=True)

    # Store the latest edited inventory for the estimate tab
    if room_inv:
        st.session_state["_current_inv"] = edited_room_inventory if edited_room_inventory else room_inv
    elif "manual_items" in dir() and manual_items:
        st.session_state["_current_inv"] = {"Manual Entry": manual_items}
    else:
        st.session_state["_current_inv"] = {}


# ── Tab 2: Estimate ──────────────────────────────────────────────────
with tab_estimate:
    st.subheader("Moving Cost Estimate")

    # Get current inventory from the Inventory tab
    current_inventory = st.session_state.get("_current_inv", st.session_state.room_inventory)

    combined = merge_room_inventories(current_inventory)

    if combined:
        summary = summarise_items(combined)
        quote_data = calculate_quote(
            summary, origin, destination, int(distance_km),
            floor_number=int(floor_number), has_lift=has_lift,
            extra_boxes=int(extra_boxes), box_type=box_type,
        )

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🚛 Truck", quote_data["truck"])
        m2.metric("📦 Volume", f"{quote_data['total_volume']} cu ft")
        m3.metric("⚖️ Weight", f"{quote_data['total_weight']} kg")
        m4.metric("⚠️ Fragile", quote_data["fragile_items"])

        st.divider()

        # Quote range
        q1, q2, q3 = st.columns(3)
        q1.metric("Min Estimate", f"INR {quote_data['lower_estimate']:,}")
        q2.metric("Best Estimate", f"INR {quote_data['midpoint']:,}")
        q3.metric("Max Estimate", f"INR {quote_data['upper_estimate']:,}")

        st.divider()

        # Cost breakdown
        col_cost, col_pdf = st.columns([1.5, 1])

        with col_cost:
            st.markdown("**Cost Breakdown**")
            costs = pd.DataFrame([
                {"Cost Head": "Base truck", "Amount (INR)": f"{quote_data['base_cost']:,}"},
                {"Cost Head": "Transport", "Amount (INR)": f"{quote_data['transport_cost']:,}"},
                {"Cost Head": "Packing material", "Amount (INR)": f"{quote_data['packing_cost']:,}"},
                {"Cost Head": "Loading & unloading", "Amount (INR)": f"{quote_data['loading_cost']:,}"},
                {"Cost Head": "Fragile surcharge", "Amount (INR)": f"{quote_data['fragile_cost']:,}"},
                {"Cost Head": "Floor surcharge", "Amount (INR)": f"{quote_data['floor_cost']:,}"},
                {"Cost Head": "Extra boxes", "Amount (INR)": f"{quote_data['box_cost']:,}"},
            ])
            st.dataframe(costs, width="stretch", hide_index=True)

        with col_pdf:
            st.markdown("**Download Report**")
            pdf_path = generate_report_pdf(
                current_inventory, summary, quote_data,
                room_frames=st.session_state.room_frames,
            )
            st.download_button(
                "📄 Download PDF Report",
                data=pdf_path.read_bytes(),
                file_name=f"MoveVision_Estimate_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                width="stretch",
            )
            st.caption("Includes room-wise inventory, cost breakdown, packing tips, and terms.")

        # Packing tips
        st.divider()
        st.markdown("**📝 Packing Tips for Your Items**")
        tips_shown = 0
        for item_name in sorted(combined.keys()):
            if item_name in PACKING_TIPS:
                st.markdown(f"- **{item_name.title()}**: {PACKING_TIPS[item_name]}")
                tips_shown += 1
        if tips_shown == 0:
            st.caption("No specific packing tips for detected items.")
    else:
        st.info("Scan room photos or add inventory items to generate an estimate.")


# ── Tab 3: Annotated Frames ─────────────────────────────────────────
with tab_frames:
    st.subheader("Detection Frames")
    room_frames = st.session_state.room_frames

    if room_frames:
        for room_name in sorted(room_frames.keys()):
            frames = room_frames[room_name]
            if not frames:
                continue
            with st.expander(f"🏠 {room_name}  —  {len(frames)} frame(s)", expanded=True):
                cols = st.columns(min(3, len(frames)))
                for idx, frame in enumerate(frames[:3]):
                    cols[idx % len(cols)].image(frame_to_rgb(frame), width="stretch")
    else:
        st.info("Detection frames will appear here after scanning.")






  
