# app.py
import os
import json
import io
import traceback
from typing import Optional

import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

# Try to import Gemini client
try:
    import google.generativeai as genai
except Exception as e:
    genai = None

# ----------------------
# Config & Helpers
# ----------------------
load_dotenv()  # loads .env if present
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

st.set_page_config(page_title="Excel Deep Analysis (Local • Gemini)", layout="wide")
st.title("📊 Excel Deep Analysis — Local (Gemini)")

if genai is None:
    st.error("Missing google.generativeai package. Install with:\n\npip install google-generativeai")
    st.stop()

if not GEMINI_API_KEY:
    st.warning("No GEMINI_API_KEY found in environment. Create a .env with GEMINI_API_KEY=your_key or set env var.")
    st.info("If you already have a key, put it in a .env file or set environment variable and restart the app.")
    # continue: user may still want to upload but model won't call

else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.error(f"Failed to configure Gemini key: {e}")
        st.stop()

# Utility: attempt to parse model response robustly
def extract_text_from_response(resp) -> str:
    """
    Try common places to extract response text from various forms of gemini response object.
    """
    try:
        if hasattr(resp, "text"):
            return resp.text
        # older variants
        if hasattr(resp, "candidates") and len(resp.candidates) > 0:
            c = resp.candidates[0]
            if hasattr(c, "content"):
                return str(c.content)
            return str(c)
        return str(resp)
    except Exception:
        return str(resp)

# Render table without index and center-align integers
def show_dataframe_centered(df: pd.DataFrame):
    # Round numeric columns to 0 decimals if they are effectively integers
    df_display = df.copy()
    for col in df_display.select_dtypes(include="number").columns:
        # round values that are near-integers
        df_display[col] = df_display[col].round(0).astype(pd.Int64Dtype())
    st.dataframe(df_display.style.set_properties(**{"text-align":"center"}), use_container_width=True)

# Parse JSON safely
def parse_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        # try to extract JSON substring between first { and last }
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except Exception:
            return None

# ----------------------
# UI — File upload
# ----------------------
uploaded_file = st.file_uploader("Upload Excel (.xlsx) or CSV (.csv)", type=["csv", "xlsx"])
sample_df = None
if uploaded_file:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            sample_df = pd.read_csv(uploaded_file)
        else:
            sample_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    st.success(f"Loaded `{uploaded_file.name}` — {sample_df.shape[0]} rows × {sample_df.shape[1]} cols")
    st.write("### Preview")
    st.dataframe(sample_df.head(10), use_container_width=True)

    # show basic column types & sample values
    with st.expander("Columns & types"):
        cols_info = pd.DataFrame({
            "column": sample_df.columns,
            "dtype": [str(sample_df[c].dtype) for c in sample_df.columns],
            "non_null": [int(sample_df[c].notna().sum()) for c in sample_df.columns]
        })
        st.dataframe(cols_info, use_container_width=True)

# ----------------------
# UI — Question input
# ----------------------
question = st.text_input("Ask a question about your data (e.g., 'Department wise % increased 23-24', 'age bins of 5 years and headcount bar chart')")

run_auto = st.checkbox("Auto-run on typing (runs whenever you type)", value=False)
analyze_clicked = st.button("Analyze")

should_run = False
if uploaded_file and sample_df is not None:
    if analyze_clicked:
        should_run = True
    elif run_auto and question.strip():
        should_run = True

if not uploaded_file:
    st.info("Upload a file to enable analysis.")
    should_run = False

# ----------------------
# Analysis runner
# ----------------------
if should_run:
    with st.spinner("Asking Gemini and analyzing..."):
        # Build a controlled prompt requesting JSON output
        col_list = list(sample_df.columns)
        # small sample to include but limit size
        head_csv = sample_df.head(10).to_csv(index=False)

        prompt = f"""
You are an expert data analyst. You will be provided:
1) A list of dataframe columns (name and dtype).
2) A small CSV sample of the dataframe (first 10 rows).
3) A user question.

Your task: answer the question precisely using Pandas/analysis logic. RETURN A SINGLE JSON OBJECT ONLY (no additional commentary).
The JSON must be valid and strictly follow this schema:

{{
  "type": "table" | "chart" | "text",            # indicates output type
  "table_csv": "<CSV string>" | null,            # if type == table, return CSV text (no index)
  "chart_spec": {{                               
      "kind":"bar"|"line"|"pie"|"histogram",      # chart type
      "x":"column_name",
      "y":"column_name",
      "aggregate":"sum"|"mean"|"count"|"none"
  }} | null,
  "text": "<short textual conclusion, numbers rounded to integers>"  # small textual summary
}}

Rules:
- If the user asked for a chart (bar, histogram, etc.), return type="chart" and populate chart_spec (and you may also include table_csv).
- If a table is requested or natural, return type="table" and put the results as CSV string in table_csv.
- All numeric values in table_csv should be rounded to integers (no decimals).
- The "text" field should be 1-2 sentences summarizing the key result (rounded numbers).
- Use the columns exactly as provided. Do NOT invent new columns.
- If you cannot compute because of missing columns, return type="text" with an explanation in "text".
- USER QUESTION: {question}

COLUMNS: {json.dumps([{ "name": c, "dtype": str(sample_df[c].dtype) } for c in col_list])}
SAMPLE_CSV:
{head_csv}
"""

        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content([prompt])
            resp_text = extract_text_from_response(response)
        except Exception as e:
            st.error("Error calling Gemini API:")
            st.text(traceback.format_exc())
            resp_text = None

        if not resp_text:
            st.error("No response from model.")
        else:
            # Try parse JSON
            parsed = parse_json(resp_text)
            if parsed is None:
                st.warning("Model did not return strict JSON. Showing raw text output. You can try rephrasing the question.")
                st.subheader("Raw model output")
                st.text(resp_text)
            else:
                # Process structured output
                out_type = parsed.get("type", "text")
                st.subheader("Model conclusion")
                st.write(parsed.get("text", ""))

                if out_type == "table" and parsed.get("table_csv"):
                    try:
                        table_csv = parsed["table_csv"]
                        df_out = pd.read_csv(io.StringIO(table_csv))
                        st.subheader("Result Table")
                        show_dataframe_centered(df_out)
                    except Exception as e:
                        st.error("Failed to parse table CSV from model output.")
                        st.text(str(e))

                if out_type == "chart" and parsed.get("chart_spec"):
                    spec = parsed["chart_spec"]
                    kind = spec.get("kind")
                    x = spec.get("x")
                    y = spec.get("y")
                    agg = spec.get("aggregate", "none")

                    if x not in sample_df.columns:
                        st.error(f"Column {x} missing from dataset.")
                    else:
                        # prepare data for plot
                        plot_df = sample_df.copy()
                        if y and y in plot_df.columns and agg != "none":
                            if agg == "sum":
                                grouped = plot_df.groupby(x)[y].sum().reset_index()
                            elif agg == "mean":
                                grouped = plot_df.groupby(x)[y].mean().reset_index()
                            elif agg == "count":
                                grouped = plot_df.groupby(x)[y].count().reset_index()
                            else:
                                grouped = plot_df[[x, y]].copy()
                            df_plot = grouped
                            # round numerical values
                            for col in df_plot.select_dtypes(include="number").columns:
                                df_plot[col] = df_plot[col].round(0)
                        else:
                            df_plot = plot_df

                        st.subheader("Result Chart")
                        try:
                            if kind == "bar":
                                fig = px.bar(df_plot, x=x, y=y)
                            elif kind == "line":
                                fig = px.line(df_plot, x=x, y=y)
                            elif kind == "pie":
                                # pie requires names and values
                                fig = px.pie(df_plot, names=x, values=y)
                            elif kind == "histogram":
                                fig = px.histogram(df_plot, x=x)
                            else:
                                st.info("Unknown chart kind. Showing a table instead.")
                                show_dataframe_centered(df_plot.head(50))
                                fig = None
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error("Failed to generate chart from spec.")
                            st.text(str(e))

                if out_type == "text" and parsed.get("text"):
                    st.info(parsed["text"])

    # cleanup: ensure no file saved to disk (we never wrote one)
    st.write("")  # place holder

# End of app.py
