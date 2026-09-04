from __future__ import annotations

import runpy
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BRAND_DIR = ROOT / "assets" / "brand"
APP_ICON = BRAND_DIR / "courtflux-app-icon.webp"
HORIZONTAL_LOGO = BRAND_DIR / "courtflux-horizontal-dark.webp"

# Keep the benchmark implementation isolated in beta_app.py. This entrypoint only
# applies CourtFlux visual identity before executing the existing beta unchanged.
_original_set_page_config = st.set_page_config
_original_title = st.title


def _courtflux_page_config(*args, **kwargs):
    kwargs["page_title"] = "CourtFlux — Basketball Intelligence from Video"
    if APP_ICON.exists():
        with Image.open(APP_ICON) as image:
            kwargs["page_icon"] = image.copy()
    return _original_set_page_config(*args, **kwargs)


def _courtflux_title(body, *args, **kwargs):
    if "CourtFlux" in str(body) and HORIZONTAL_LOGO.exists():
        st.image(str(HORIZONTAL_LOGO), width=600)
        return None
    return _original_title(body, *args, **kwargs)


st.set_page_config = _courtflux_page_config
st.title = _courtflux_title

runpy.run_path(str(Path(__file__).with_name("beta_app.py")), run_name="__main__")
