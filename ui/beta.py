from __future__ import annotations

import runpy
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BRAND_DIR = ROOT / "assets" / "brand"
APP_ICON = BRAND_DIR / "courtflux-app-icon.webp"
HORIZONTAL_LOGO = BRAND_DIR / "courtflux-horizontal-dark.webp"

# Public marketing page and internal benchmark intentionally share one entrypoint.
# The existing benchmark remains isolated in beta_app.py and is not restructured.
_original_set_page_config = st.set_page_config
_original_title = st.title


def _page_icon():
    if not APP_ICON.exists():
        return "🏀"
    with Image.open(APP_ICON) as image:
        return image.copy()


def _courtflux_page_config(*args, **kwargs):
    kwargs["page_title"] = "CourtFlux — Basketball Intelligence from Video"
    kwargs["page_icon"] = _page_icon()
    return _original_set_page_config(*args, **kwargs)


def _courtflux_title(body, *args, **kwargs):
    # Keep the benchmark controls/layout intact; only replace the existing title
    # with the approved CourtFlux horizontal lockup.
    if "CourtFlux" in str(body) and HORIZONTAL_LOGO.exists():
        st.image(str(HORIZONTAL_LOGO), width=600)
        return None
    return _original_title(body, *args, **kwargs)


def _current_view() -> str:
    try:
        value = st.query_params.get("view", "site")
        if isinstance(value, list):
            value = value[0] if value else "site"
        return str(value).lower()
    except Exception:
        try:
            params = st.experimental_get_query_params()
            return str((params.get("view") or ["site"])[0]).lower()
        except Exception:
            return "site"


if _current_view() == "lab":
    # This executes the existing test interface unchanged. No marketing navigation,
    # sections, cards or extra controls are injected into beta_app.py.
    st.set_page_config = _courtflux_page_config
    st.title = _courtflux_title
    runpy.run_path(str(Path(__file__).with_name("beta_app.py")), run_name="__main__")
else:
    _original_set_page_config(
        page_title="CourtFlux — Basketball Intelligence from Video",
        page_icon=_page_icon(),
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    runpy.run_path(str(Path(__file__).with_name("marketing.py")), run_name="__main__")
