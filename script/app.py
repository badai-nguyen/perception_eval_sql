import streamlit as st

st.set_page_config(
    page_title="Evaluation Dashboard",
    layout="wide",
)

st.title("Evaluation Dashboard")
mode = st.sidebar.radio(
    "Mode",
    ["Single Run", "Compare Runs"],
)


# default paths
DEFAULT_A = "data/Summary.csv"
DEFAULT_B = "data2/Summary.csv"




st.markdown("""
Use the sidebar to switch pages:

- **Tracking Stats**: position / velocity metrics  
- **Criteria Evaluation**: criteria-based performance metrics
""")
