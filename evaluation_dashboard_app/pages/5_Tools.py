import streamlit as st
import re
import subprocess

from lib.page_chrome import inject_app_page_styles, render_page_hero

st.set_page_config(
    page_title="lsim_analysis_tool runner",
    page_icon="⚙️",
    layout="centered",
)
inject_app_page_styles()
render_page_hero(
    kicker="CLI bridge",
    title="lsim_analysis_tool runner",
    description=(
        "Paste Autoware Evaluator report or suite URLs, generate shell snippets, and run analysis commands "
        "from a simple form."
    ),
    mode="Single Run",
)

# Constants and regexes
JOB_RE = re.compile(r"/reports/([0-9a-fA-F-]{36})")
SUITE_RE = re.compile(r"/suites/([0-9a-fA-F-]{36})")
DEFAULT_REPORT_URL = (
    "https://evaluation.tier4.jp/evaluation/reports/"
    "71b8eec9-7e28-5f9c-9b89-8e88545e742f?project_id=x2_dev"
)
DEFAULT_SUITE_URL = (
    "https://evaluation.tier4.jp/evaluation/suites/"
    "1af11feb-362d-4c48-b258-02cd433a3866?project_id=x2_dev"
)
DEFAULT_OUTPUT = "~/data/x2gen2/evaluator_summary/NO_shorten_left_lower_gpu2_No3/"

def extract_job_id(report_url):
    m = JOB_RE.search(report_url or "")
    return m.group(1) if m else ""

def extract_suite_id(suite_url):
    m = SUITE_RE.search(suite_url or "")
    return m.group(1) if m else ""

# App state initialization
if 'report_url' not in st.session_state:
    st.session_state['report_url'] = DEFAULT_REPORT_URL
if 'suite_url' not in st.session_state:
    st.session_state['suite_url'] = DEFAULT_SUITE_URL

# Layout inputs
with st.form(key="eval_runner_form"):
    col1, col2 = st.columns([1, 1])
    with col1:
        project_id = st.text_input("Project ID", value="x2_dev", key="project_id")
        setup_bash = st.text_area(
            "setup.bash path",
            value="/home/leigu/pilot-auto.x2.v4.3/install/setup.bash",
            key="setup_bash",
            height=120,
            placeholder="Enter full path(s) to your setup.bash file(s), one per line."
        )
        output_dir = st.text_area(
            "Output Directory",
            value=DEFAULT_OUTPUT,
            key="output_dir",
            height=120,
            placeholder="Enter one or more output directories, one per line."
        )
    with col2:
        report_url = st.text_area(
            "Report URL", 
            value=st.session_state['report_url'], 
            key="report_url",
            height=120,
            placeholder="Paste the full Evaluation Report URL here."
        )
        suite_url = st.text_area(
            "Suite URL", 
            value=st.session_state['suite_url'], 
            key="suite_url",
            height=120,
            placeholder="Paste the full Evaluation Suite URL here."
        )

        # Job ID and Suite ID auto-extracted from URL text fields live as you type
        # So always extract from form inputs (not session state nor callbacks)
        job_id = extract_job_id(report_url)
        suite_id = extract_suite_id(suite_url)

        st.text_input("Job ID", value=job_id, key="job_id", disabled=True)
        st.text_input("Suite ID", value=suite_id, key="suite_id", disabled=True)

    # Build command
    cmd = (
        f"./perception_evaluation_result_creator2.sh "
        f"{setup_bash} "
        f"./perception_eval_result_summarizer.py "
        f"{project_id} "
        f"{job_id} "
        f"{suite_id} "
        f"{output_dir}"
    )

    # Submit button as required for Streamlit forms
    submitted = st.form_submit_button("Run in Terminal")

# "Run in Terminal" logic
if submitted:
    st.info(f"Command to run (copy below and paste into your terminal):\n\n{cmd}")


st.markdown("""
---
**Instructions:**
- Enter your parameters above.
- Job ID / Suite ID are automatically parsed when you enter the Evaluation URLs.
- Click **Run in Terminal** to show the command for copy-paste.
""")