
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Business Entity Resolution",
    page_icon="🔎",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent

st.title("🔎 Business Entity Resolution")
st.caption("Amazon ML Challenge | Entity Matching Dashboard")

st.markdown(
    """
    This application helps explore business entity matching results
    across multiple data sources.
    """
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Project", "Entity Resolution")

with col2:
    st.metric("Sources", "3")

with col3:
    st.metric("Status", "Demo")

st.subheader("Explore Matching Results")

results_file = ROOT / "output" / "matching_results.tsv"

if results_file.exists():
    st.success("Matching results file found.")

    if st.button("Load sample results"):
        import pandas as pd

        df = pd.read_csv(results_file, sep="\t", nrows=100)
        st.dataframe(df, use_container_width=True)
else:
    st.warning(
        "Matching results are not included in this deployment yet. "
        "Configure the results file and original source datasets "
        "to enable business-name lookup."
    )

st.divider()
st.caption("Business Entity Resolution Project")
