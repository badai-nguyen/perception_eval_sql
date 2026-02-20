import streamlit as st
from pathlib import Path
import re

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