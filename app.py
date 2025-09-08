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
            q = query.lower()
            computed_answer = None
            chart = None

            # ---------- pandas computations ----------
            try:
                # TOTAL / SUM
                if "total" in q or "sum" in q:
                    words = q.split()
                    target_col = None
                    for w in words:
                        col_match = match_column(w, df.columns)
                        if col_match:
                            target_col = col_match
                            break

                    if target_col:
                        total_val = pd.to_numeric(df[target_col], errors="coerce").sum()
                        computed_answer = pd.DataFrame({"Column": [target_col], "Total": [total_val]})
                        st.markdown(f"### 📝 Total of {target_col}")
                        st.write(computed_answer)
                        download_button(computed_answer, "total_results.csv")
                    else:
                        numeric_cols = df.select_dtypes(include="number").columns
                        if len(numeric_cols) > 0:
                            computed_answer = df[numeric_cols].sum().to_frame("Total").reset_index()
                            st.markdown("### 📝 Totals from All Numeric Columns")
                            st.write(computed_answer)
                            download_button(computed_answer, "total_results.csv")

                # AVERAGE / MEAN
                elif "average" in q or "mean" in q:
                    words = q.split()
                    target_col = None
                    for w in words:
                        col_match = match_column(w, df.columns)
                        if col_match:
                            target_col = col_match
                            break

                    if target_col:
                        avg_val = pd.to_numeric(df[target_col], errors="coerce").mean()
                        computed_answer = pd.DataFrame({"Column": [target_col], "Average": [avg_val]})
                        st.markdown(f"### 📝 Average of {target_col}")
                        st.write(computed_answer)
                        download_button(computed_answer, "average_results.csv")
                    else:
                        numeric_cols = df.select_dtypes(include="number").columns
                        if len(numeric_cols) > 0:
                            computed_answer = df[numeric_cols].mean().to_frame("Average").reset_index()
                            st.markdown("### 📝 Averages from All Numeric Columns")
                            st.write(computed_answer)
                            download_button(computed_answer, "average_results.csv")

                # GROUP BY (improved prioritization)
                elif "group" in q or " by " in q or "wise" in q:
                    words = q.replace("wise", "by").split()
                    group_candidate = None
                    value_candidate = None

                    for w in words:
                        col_match = match_column(w, df.columns)
                        if col_match:
                            low = col_match.lower()
                            # Prioritize grouping columns
                            if any(k in low for k in ["name", "dept", "category", "type", "group", "region", "center", "cost center", "cost_center", "account"]):
                                group_candidate = col_match
                            # Prioritize numeric/value columns
                            elif any(k in low for k in ["amount", "value", "price", "cost", "lc", "total", "amt"]):
                                value_candidate = col_match
                            else:
                                if group_candidate is None:
                                    group_candidate = col_match
                                else:
                                    value_candidate = col_match

                    # fallback numeric column
                    if value_candidate is None:
                        numeric_cols = df.select_dtypes(include="number").columns
                        if len(numeric_cols) > 0:
                            value_candidate = numeric_cols[0]

                    if group_candidate and value_candidate:
                        df[value_candidate] = pd.to_numeric(df[value_candidate], errors="coerce")
                        computed_answer = df.groupby(group_candidate)[value_candidate].sum().reset_index()
                        st.markdown(f"### 📝 {value_candidate} by {group_candidate}")
                        st.write(computed_answer)
                        download_button(computed_answer, "groupby_results.csv")
                        chart = px.bar(computed_answer, x=group_candidate, y=value_candidate,
                                       title=f"{value_candidate} by {group_candidate}")

                # GRAPH/CHART REQUESTS
                elif any(k in q for k in ["chart", "graph", "bar", "pie", "line", "hist", "histogram"]):
                    numeric_cols = df.select_dtypes(include="number").columns
                    if len(numeric_cols) == 0:
                        st.warning("No numeric columns available for charting.")
                    else:
                        # Detect x and y intelligently
                        x_col = None
                        for w in q.split():
                            m = match_column(w, df.columns)
                            if m and m not in numeric_cols:
                                x_col = m
                                break
                        if x_col is None:
                            x_col = df.columns[0]

                        y_col = None
                        for w in q.split():
                            m = match_column(w, df.columns)
                            if m and m in numeric_cols:
                                y_col = m
                                break
                        if y_col is None:
                            y_col = numeric_cols[0]

                        if "bar" in q:
                            chart = px.bar(df, x=x_col, y=y_col, title=f"Bar: {y_col} by {x_col}")
                        elif "pie" in q:
                            chart = px.pie(df, names=x_col, values=y_col, title=f"Pie: {y_col} by {x_col}")
                        elif "line" in q:
                            chart = px.line(df, x=x_col, y=y_col, title=f"Line: {y_col} by {x_col}")
                        else:
                            chart = px.histogram(df, x=y_col, title=f"Histogram of {y_col}")

                        st.plotly_chart(chart, use_container_width=True)

            except Exception as e:
                st.warning(f"Couldn't compute directly: {e}")

            # ---------- AI Explanation via Gemini (robust handling) ----------
            if use_ai and API_KEY:
                explanation = None
                # Build short prompt for explanation
                prompt = f"""You are a data analyst. The dataframe has {len(df)} rows and {len(df.columns)} columns.
Columns: {list(df.columns)}
Query: {query}
Show a short explanation/insight only. Computed results (if any) are already shown above."""

                # Try modern GenerativeModel API first (recommended)
                try:
                    import google.generativeai as genai
                except Exception as e:
                    st.error(f"google-generativeai library not available or import failed: {e}")
                    explanation = None
                else:
                    last_err = None
                    try:
                        genai.configure(api_key=API_KEY)
                    except Exception as e:
                        st.error(f"Failed to configure google.generativeai with provided key: {e}")
                        explanation = None
                    else:
                        mname = model_name.strip() or "gemini-2.5-flash"
                        # 1) Try GenerativeModel.generate_content (modern)
                        try:
                            if hasattr(genai, "GenerativeModel"):
                                model_obj = genai.GenerativeModel(mname)
                                # generate_content sometimes expects keyword 'prompt' or 'input' depending on SDK; try both
                                try:
                                    resp = model_obj.generate_content(prompt)
                                except TypeError:
                                    # try input=...
                                    resp = model_obj.generate_content(input=prompt)
                                # resp may have .text or other structure
                                if hasattr(resp, "text"):
                                    explanation = resp.text
                                else:
                                    # try to stringify
                                    explanation = str(resp)
                        except Exception as e:
                            last_err = e
                            explanation = None

                        # 2) Fallback: genai.generate_text (older SDKs)
                        if not explanation:
                            try:
                                if hasattr(genai, "generate_text"):
                                    resp = genai.generate_text(model=mname, prompt=prompt)
                                    if hasattr(resp, "text"):
                                        explanation = resp.text
                                    elif hasattr(resp, "candidates") and len(resp.candidates) > 0:
                                        # some variants
                                        cand = resp.candidates[0]
                                        if hasattr(cand, "content"):
                                            explanation = cand.content
                                        else:
                                            explanation = str(cand)
                                    else:
                                        explanation = str(resp)
                            except Exception as e:
                                last_err = e
                                explanation = None

                        # 3) Fallback: genai.models.generate(...)
                        if not explanation:
                            try:
                                if hasattr(genai, "models") and hasattr(genai.models, "generate"):
                                    resp = genai.models.generate(model=mname, prompt=prompt)
                                    # resp may contain 'output' or similar
                                    if hasattr(resp, "output"):
                                        explanation = str(resp.output)
                                    else:
                                        explanation = str(resp)
                            except Exception as e:
                                last_err = e
                                explanation = None

                        if not explanation and last_err:
                            st.error("Could not extract text from Google SDK. Last SDK error: " + str(last_err))

                if explanation:
                    st.markdown("### 🤖 AI Explanation (Gemini)")
                    st.write(explanation)

            # ---------- Show chart if computed earlier ----------
            if chart:
                st.plotly_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"Error reading file: {e}")

else:
    if not uploaded_file:
        st.info("Upload a CSV or Excel file to get started.")
    if use_ai and not API_KEY:
        st.warning("Provide Google API key in Streamlit Secrets (API_KEY) or in the sidebar to enable AI explanations.")
