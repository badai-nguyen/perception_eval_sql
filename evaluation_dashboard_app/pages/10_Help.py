import streamlit as st
from pathlib import Path
import re

from lib.page_chrome import inject_app_page_styles, render_page_hero

st.set_page_config(
    page_title="Help",
    page_icon="❔",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_app_page_styles()
render_page_hero(
    kicker="Documentation",
    title="Help & guide",
    description="In-app copy of the project README — setup, pages, and workflows for the evaluation dashboard.",
    mode="Single Run",
)

readme_path = Path("Readme.md")
content = readme_path.read_text(encoding="utf-8")

# Find markdown images
image_pattern = r"!\[(.*?)\]\((.*?)\)"

parts = re.split(image_pattern, content)

i = 0
while i < len(parts):
    st.markdown(parts[i])
    
    if i + 2 < len(parts):
        alt_text = parts[i + 1]
        img_path = parts[i + 2]

        img_file = Path(img_path)
        if img_file.exists():
            st.image(str(img_file), caption=alt_text)
        else:
            st.warning(f"Image not found: {img_path}")

        i += 3
    else:
        break