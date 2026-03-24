import json
import re
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

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

# Streamlit markdown does not run Mermaid; split fenced ```mermaid blocks and render via Mermaid.js.
MERMAID_FENCE = re.compile(r"```mermaid\s*\n([\s\S]*?)```", re.IGNORECASE)
IMAGE_PATTERN = re.compile(r"!\[(.*?)\]\((.*?)\)")


def _render_mermaid(definition: str) -> None:
    """Render a Mermaid diagram inside an HTML component (CDN script)."""
    defn_json = json.dumps(definition.strip())
    uid = uuid.uuid4().hex[:12]
    html = f"""
<div id="mermaid-host-{uid}" style="overflow:auto;max-width:100%;padding:0.25rem 0;"></div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.0/dist/mermaid.min.js"></script>
<script>
(function() {{
  const defn = {defn_json};
  const host = document.getElementById("mermaid-host-{uid}");
  mermaid.initialize({{ startOnLoad: false, theme: "neutral", securityLevel: "loose" }});
  const graphId = "mermaid-graph-{uid}";
  mermaid.render(graphId, defn).then(function(res) {{
    host.innerHTML = res.svg;
  }}).catch(function(err) {{
    host.textContent = "Mermaid diagram could not be rendered: " + String(err);
  }});
}})();
</script>
"""
    components.html(html, height=480, scrolling=True)


def _render_markdown_with_images(chunk: str) -> None:
    parts = IMAGE_PATTERN.split(chunk)
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


readme_path = Path("Readme.md")
content = readme_path.read_text(encoding="utf-8")

for idx, piece in enumerate(MERMAID_FENCE.split(content)):
    if idx % 2 == 0:
        _render_markdown_with_images(piece)
    else:
        _render_mermaid(piece)
