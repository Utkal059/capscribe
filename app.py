"""
app.py — CapScribe Streamlit Web UI
Run with: streamlit run app.py
Deploy free at: https://streamlit.io/cloud
"""

import io
import json
import tempfile
import os
from pathlib import Path

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CapScribe",
    page_icon="📄",
    layout="wide",
)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("📄 CapScribe")
st.markdown(
    "**Structured capital event extraction from DRHP / IPO filings using Claude AI.**  \n"
    "Upload a PDF prospectus and get a clean JSON + CSV of all allotments, "
    "bonus issues, rights issues, and authorised capital changes."
)

st.divider()

# ── Sidebar config ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Your Anthropic API key. Never stored or logged.",
    )
    chunk_size = st.slider(
        "Pages per chunk",
        min_value=5, max_value=60, value=20, step=5,
        help="Smaller = cheaper per call but more calls. 20 is a good default.",
    )
    model = st.selectbox(
        "Model",
        options=[
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-6",
        ],
        help="Haiku is ~4x cheaper than Sonnet. Use Haiku for cost efficiency.",
    )
    force_rerun = st.checkbox(
        "Force re-run (ignore cache)",
        value=False,
        help="Re-process all chunks even if cached results exist.",
    )
    st.divider()
    st.caption("CapScribe · MIT License · [GitHub](https://github.com/Utkal059/capscribe)")

# ── File upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload DRHP / IPO Prospectus PDF",
    type=["pdf"],
    help="DRHPs, RHPs, and AGM notices are supported.",
)

if uploaded_file and not api_key:
    st.warning("⚠️ Please enter your Anthropic API key in the sidebar to proceed.")

if uploaded_file and api_key:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info(f"**File:** {uploaded_file.name}  |  **Size:** {uploaded_file.size / 1024:.1f} KB")
    with col2:
        run = st.button("🚀 Extract Capital Events", type="primary", use_container_width=True)

    if run:
        # Set env vars for extractor
        os.environ["ANTHROPIC_API_KEY"] = api_key
        os.environ["CHUNK_SIZE"] = str(chunk_size)
        os.environ["CAPSCRIBE_MODEL"] = model

        # Import here so env vars are set before module-level config runs
        try:
            from extractor import run_extraction
        except ImportError as e:
            st.error(f"Import error: {e}. Make sure you're running from the capscribe directory.")
            st.stop()

        # Save uploaded file to a temp path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Run extraction with progress feedback
        progress_bar = st.progress(0, text="Initialising...")
        status = st.empty()

        try:
            status.info("📖 Reading PDF and splitting into chunks...")
            progress_bar.progress(10, text="Reading PDF...")

            result = run_extraction(tmp_path, force_rerun=force_rerun)

            progress_bar.progress(100, text="Done!")
            status.success(
                f"✅ Extracted **{result['total_events']} capital events** "
                f"from {result['total_pages']} pages.  "
                f"Estimated cost: **${result.get('estimated_cost_usd', 0):.4f} USD**"
            )

        except Exception as e:
            progress_bar.empty()
            st.error(f"❌ Extraction failed: {e}")
            st.stop()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        events = result.get("capital_events", [])

        if not events:
            st.warning("No capital events found in this document. Check the PDF is text-based (not scanned).")
        else:
            # ── Summary metrics ───────────────────────────────────────────────
            st.subheader("📊 Summary")
            by_type = {}
            for ev in events:
                t = ev.get("event_type", "unknown")
                by_type[t] = by_type.get(t, 0) + 1

            cols = st.columns(len(by_type) + 1)
            cols[0].metric("Total Events", len(events))
            for i, (etype, count) in enumerate(by_type.items()):
                label = etype.replace("_", " ").title()
                cols[i + 1].metric(label, count)

            low_conf = [e for e in events if e.get("confidence", 1) < 0.5]
            if low_conf:
                st.warning(
                    f"⚠️ {len(low_conf)} event(s) have low confidence (<0.5). "
                    "These may have incomplete data — review manually."
                )

            # ── Data table ────────────────────────────────────────────────────
            st.subheader("📋 Extracted Events")
            import pandas as pd
            df = pd.DataFrame(events)
            # Reorder columns for readability
            priority_cols = ["event_type", "date", "shares", "face_value", "issue_price",
                             "consideration", "allottee_category", "ratio", "shares_issued",
                             "pre_issue_capital", "post_issue_capital",
                             "old_capital", "new_capital", "resolution_type",
                             "confidence", "source_pages"]
            existing_priority = [c for c in priority_cols if c in df.columns]
            other_cols = [c for c in df.columns if c not in priority_cols]
            df = df[existing_priority + other_cols]

            # Colour-code confidence
            def colour_confidence(val):
                if not isinstance(val, (int, float)):
                    return ""
                if val >= 0.8:
                    return "background-color: #d4edda"
                if val >= 0.5:
                    return "background-color: #fff3cd"
                return "background-color: #f8d7da"

            styled = df.style.applymap(colour_confidence, subset=["confidence"] if "confidence" in df.columns else [])
            st.dataframe(styled, use_container_width=True, height=400)

            # ── Downloads ─────────────────────────────────────────────────────
            st.subheader("⬇️ Download")
            dl_col1, dl_col2 = st.columns(2)

            json_bytes = json.dumps(result, indent=2, ensure_ascii=False).encode("utf-8")
            dl_col1.download_button(
                label="📥 Download JSON",
                data=json_bytes,
                file_name=f"{Path(uploaded_file.name).stem}_extracted.json",
                mime="application/json",
                use_container_width=True,
            )

            # Build CSV in memory
            import csv
            csv_buf = io.StringIO()
            if events:
                fieldnames = list(df.columns)
                writer = csv.DictWriter(csv_buf, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for ev in events:
                    writer.writerow({k: ev.get(k, "") for k in fieldnames})
            dl_col2.download_button(
                label="📥 Download CSV",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name=f"{Path(uploaded_file.name).stem}_extracted.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ── Empty state ───────────────────────────────────────────────────────────────
else:
    st.markdown(
        """
        ### How it works
        1. Enter your Anthropic API key in the sidebar
        2. Upload a DRHP or IPO prospectus PDF
        3. Click **Extract Capital Events**
        4. Download the structured JSON or CSV

        ### What gets extracted
        | Event Type | Fields |
        |---|---|
        | **Allotment** | Date, shares, face value, issue price, consideration, allottee |
        | **Bonus Issue** | Date, ratio, shares issued, pre/post capital |
        | **Rights Issue** | Date, ratio, price, shares offered |
        | **Authorised Capital Change** | Date, old capital, new capital, resolution type |

        > 💡 **Cost tip:** Use Haiku model + chunk size 20 for maximum cost efficiency.
        """
    )
