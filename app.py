# app.py
import os
import json
import re
import textwrap
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()

# --- Gemini client (google-generativeai) ---
try:
    import google.generativeai as genai
except Exception:
    genai = None

st.set_page_config(page_title="Excel Deep Analysis (Gemini)", layout="wide")
st.title("📊 Excel Deep Analysis — Localhost (Gemini)")

st.markdown(
    "Upload an Excel/CSV (no files are saved). Ask questions in natural language. "
    "The model will propose safe pandas code which you can review and run locally."
)

# ---- Load API Key from ENV ----
API_KEY = os.getenv("GENIE_API_KEY")
if not API_KEY:
    st.warning(
        "No Gemini API key found in environment variable `GENIE_API_KEY`. "
        "Set it before running the app. (See README / instructions.)"
    )
else:
    if genai is not None:
        genai.configure(api_key=API_KEY)

# ---- File upload (in-memory) ----
uploaded_file = st.file_uploader("Upload Excel or CSV (keeps in memory only)", type=["csv", "xlsx"])

df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success(f"Loaded `{uploaded_file.name}` — shape {df.shape}")
        st.dataframe(df.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

# ---- User question ----
question = st.text_input("Ask a question about the data (e.g. 'department wise % increase in 2023-24')")

# Optional: immediate run toggle
auto_run = st.checkbox("Auto-run generated code without confirmation (risky)", value=False)

# Helper to build the prompt sent to Gemini
def build_prompt(df_sample_csv, columns_list, user_question):
    instructions = textwrap.dedent(f"""
    You are a professional Python / Pandas data analyst.
    The dataframe variable is called `df` (pandas DataFrame).
    I will provide the dataframe columns and a small CSV sample.

    Requirements (must follow exactly):
    1) Produce a single JSON object and nothing else. The JSON must have these keys:
       - "code": a Python code string (no imports) that uses 'df' and assigns the final output
         into a variable named 'result' (Pandas DataFrame) or into 'result_figure' (matplotlib Figure).
       - "explanation": a short plain-English explanation of what the code does (<= 40 words).
    2) The code should NOT write files to disk. It must not use open(), os.system, subprocess, or any imports.
    3) Round numeric outputs to integers (use .round(0).astype(int) where relevant) so the result shows no decimals.
    4) If producing a chart, assign a matplotlib Figure to variable 'result_figure'. If producing a table, assign a DataFrame to 'result'.
    5) Keep computations deterministic and clear (groupby, agg, merge, pd.cut, etc.).
    6) Use only 'pd' and 'np' and assume they are available.
    7) Keep code reasonably short (<= 40 lines).
    8) Do not include any sensitive data in responses.

    Now, dataframe columns: {columns_list}
    CSV sample (top rows):
    {df_sample_csv}

    User question: {user_question}

    Output: single JSON object with keys 'code' and 'explanation'.
    """)
    return instructions

# Function to call Gemini model
def call_gemini(prompt_text):
    if genai is None:
        raise RuntimeError("google.generativeai package not installed.")
    model = genai.GenerativeModel("gemini-1.5-flash")  # works in many environments
    # We send prompt and sample as a single text item
    resp = model.generate_content(prompt_text)
    # response.text holds the generated text
    return resp.text if hasattr(resp, "text") else str(resp)

# Parse a JSON blob inside model output (best-effort)
def extract_json_from_text(text):
    # Find first JSON object in text
    match = re.search(r'(\{[\s\S]*\})', text)
    if not match:
        return None
    json_text = match.group(1)
    try:
        return json.loads(json_text)
    except Exception:
        # Try to clean trailing commas etc.
        cleaned = re.sub(r',\s*}', '}', json_text)
        cleaned = re.sub(r',\s*\]', ']', cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            return None

# Safe execution environment (limited builtins)
SAFE_BUILTINS = {"len": len, "min": min, "max": max, "sum": sum, "round": round, "range": range, "abs": abs}
def safe_exec(code_str, df):
    """Execute code_str in a restricted environment. Returns locals dict after exec."""
    g = {"pd": pd, "np": np, "plt": plt, "__builtins__": SAFE_BUILTINS}
    l = {"df": df.copy()}  # work on a copy to avoid accidental modifications
    exec(compile(code_str, "<string>", "exec"), g, l)
    return l

# ---- Analysis flow ----
if st.button("Analyze") or (auto_run and question.strip() and df is not None):
    if df is None:
        st.error("Upload a file first.")
    elif not API_KEY:
        st.error("Set GENIE_API_KEY environment variable before running the app.")
    else:
        # build prompt from first 8 rows CSV (keeps data small)
        sample_csv = df.head(8).to_csv(index=False)
        cols = list(df.columns)
        prompt = build_prompt(sample_csv, cols, question)
        with st.spinner("Sending question to Gemini — generating pandas code..."):
            try:
                raw = call_gemini(prompt)
            except Exception as e:
                st.error(f"Gemini call failed: {e}")
                raw = None

        if raw:
            st.subheader("Model output (raw)")
            st.code(raw)

            parsed = extract_json_from_text(raw)
            if not parsed or "code" not in parsed:
                st.error("Could not extract JSON with 'code' from the model output. Try rephrasing the question.")
            else:
                code = parsed["code"]
                explanation = parsed.get("explanation", "")
                st.subheader("Generated Python code (review before running)")
                st.code(code, language="python")
                st.markdown(f"**Explanation:** {explanation}")

                # Run automatically if auto_run True, otherwise require confirmation
                run_now = auto_run or st.button("Run generated code now")
                if run_now:
                    try:
                        locals_after = safe_exec(code, df)
                        # Check for DataFrame result
                        if "result" in locals_after and isinstance(locals_after["result"], pd.DataFrame):
                            res_df = locals_after["result"]
                            # round numeric columns and convert to int where applicable
                            num_cols = res_df.select_dtypes(include=[np.number]).columns
                            try:
                                res_df[num_cols] = res_df[num_cols].round(0).astype("Int64")
                            except Exception:
                                res_df[num_cols] = res_df[num_cols].round(0)
                            st.subheader("Result — Table")
                            st.dataframe(res_df, use_container_width=True)
                        elif "result_figure" in locals_after:
                            fig = locals_after["result_figure"]
                            st.subheader("Result — Chart")
                            st.pyplot(fig)
                        else:
                            # fallback: look for any variables that look like result
                            found = False
                            for k, v in locals_after.items():
                                if isinstance(v, pd.DataFrame):
                                    st.subheader(f"Result — Table ({k})")
                                    st.dataframe(v, use_container_width=True)
                                    found = True
                                    break
                                if hasattr(v, "savefig") or "Figure" in str(type(v)):
                                    st.subheader(f"Result — Chart ({k})")
                                    st.pyplot(v)
                                    found = True
                                    break
                            if not found:
                                st.info("Executed code but no 'result' DataFrame or 'result_figure' found. Check the generated code output above.")
                    except Exception as e:
                        st.error(f"Error while executing generated code: {e}")

# ---- Example prompts helper ----
with st.expander("Examples of useful questions"):
    st.markdown(
        """
        - Department wise % increase in salary for FY24 vs FY23  
        - Show age bins (20-25, 26-30...) and headcount bar chart  
        - Average CTC by designation and department (rounded no decimals)  
        - Cross-tab: Designation vs Rating (counts)  
        - Show top 10 employees by EP_HRS and their department  
        """
    )

# ---- Footnotes & privacy ----
st.caption(
    "Privacy: Uploaded files are processed in memory only and not saved to disk by default. "
    "However, the content you send to Gemini will go to Google servers per their terms — avoid uploading sensitive personal data if you cannot share it."
)
