import streamlit as st
import pandas as pd
import plotly.express as px
from difflib import get_close_matches
import io

# ------------------ CONFIG ------------------ #
st.set_page_config(page_title="AI Data Analyzer (Gemini)", layout="wide")
st.title("📊 Sarvesh's Data Analysis Platform — Gemini Only")

# ------------------ Sidebar: API Key & Model ------------------ #
st.sidebar.header("🔑 Google AI Studio (Gemini) Setup")
# Prefer secret stored key; fallback to manual entry (less secure)
API_KEY = st.secrets.get("API_KEY", None)
if not API_KEY:
    API_KEY = st.sidebar.text_input("Enter Google API Key (AI Studio)", type="password")

# Model selection (default recommended)
model_name = st.sidebar.text_input("Gemini model (optional)", value="gemini-2.5-flash")

# AI explanation toggle
use_ai = st.sidebar.checkbox("Show AI Explanation", value=True)

if use_ai and not API_KEY:
    st.sidebar.warning("Provide Google API key in Streamlit Secrets (API_KEY) or sidebar to enable AI explanations.")

# ------------------ Helper Functions ------------------ #
def match_column(user_word, df_columns):
    """Find the closest matching column name for a user word (case-insensitive)."""
    matches = get_close_matches(user_word.lower(), [c.lower() for c in df_columns], n=1, cutoff=0.6)
    if matches:
        for col in df_columns:
            if col.lower() == matches[0]:
                return col
    return None

def download_button(df, filename="results.csv"):
    """Create CSV download button for a DataFrame."""
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv_buffer.getvalue(),
        file_name=filename,
        mime="text/csv"
    )

# ------------------ Upload ------------------ #
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Load file
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("### Preview of Data", df.head())

        # Query input
        query = st.text_input("Ask your question in plain English and hit enter:")

        if query:
            q = query.strip()
            q_lower = q.lower()
            computed_answer = None
            chart = None

            # ---------- If query likely a column name or short column-related phrase ----------
            tokens = [t for t in q_lower.replace("_", " ").split() if t]
            is_short = len(tokens) <= 3  # short queries likely point to a column
            matched_col = None

            # try exact column name match (case-insensitive)
            for col in df.columns:
                if col.lower() == q_lower:
                    matched_col = col
                    break

            # if not exact, try match_column on each token (prioritize longer token)
            if not matched_col and is_short:
                # try to find best match among tokens (try combined tokens too)
                combined = q_lower.replace(" ", "")
                matched_col = match_column(combined, df.columns)
                if not matched_col:
                    for t in tokens:
                        mc = match_column(t, df.columns)
                        if mc:
                            matched_col = mc
                            break

            # If we have a matched column AND query does NOT contain other intent keywords,
            # show the column value counts table + chart immediately.
            intent_keywords = ["total", "sum", "average", "mean", "group", "by", "chart", "graph", "bar", "pie", "line", "percentage", "percent", "hist", "count"]
            if matched_col and not any(k in q_lower for k in intent_keywords):
                # Prepare value counts table
                vc = df[matched_col].fillna("<<MISSING>>")
                vc_counts = vc.value_counts(dropna=False).reset_index()
                vc_counts.columns = [matched_col, "count"]
                st.markdown(f"### 📝 Value counts for `{matched_col}`")
                st.write(vc_counts)
                download_button(vc_counts, f"{matched_col}_value_counts
