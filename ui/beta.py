from __future__ import annotations

import runpy
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
BRAND_DIR = ROOT / "web" / "public" / "brand"
APP_ICON = BRAND_DIR / "courtflux_app_icon_128px.png"
HORIZONTAL_LOGO = BRAND_DIR / "horizontal_logo_safe_crop.png"

# Public marketing page, customer account shell and internal benchmark share one
# entrypoint. The existing benchmark remains isolated in beta_app.py.
_original_set_page_config = st.set_page_config
_original_title = st.title


def _page_icon():
    if not APP_ICON.exists():
        return "🏀"
    try:
        with Image.open(APP_ICON) as image:
            return image.copy()
    except (UnidentifiedImageError, OSError):
        return "🏀"


def _courtflux_page_config(*args, **kwargs):
    kwargs["page_title"] = "CourtFlux — Basketball Intelligence from Video"
    kwargs["page_icon"] = _page_icon()
    return _original_set_page_config(*args, **kwargs)


def _courtflux_title(body, *args, **kwargs):
    # Keep the benchmark controls/layout intact; only replace the existing title
    # with the approved CourtFlux horizontal lockup from web/public/brand.
    if "CourtFlux" in str(body) and HORIZONTAL_LOGO.exists():
        st.image(str(HORIZONTAL_LOGO), width=560)
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


view = _current_view()

if view == "lab":
    st.set_page_config = _courtflux_page_config
    st.title = _courtflux_title
    runpy.run_path(str(Path(__file__).with_name("beta_app.py")), run_name="__main__")
elif view in {"login", "signup"}:
    _original_set_page_config(
        page_title="CourtFlux — Sign in" if view == "login" else "CourtFlux — Join the Beta",
        page_icon=_page_icon(),
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    runpy.run_path(str(Path(__file__).with_name("auth.py")), run_name="__main__")
else:
    _original_set_page_config(
        page_title="CourtFlux — Basketball Intelligence from Video",
        page_icon=_page_icon(),
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    # Streamlit's own footer is hidden by the marketing stylesheet; this higher-
    # specificity rule keeps the custom CourtFlux footer visible.
    st.markdown("<style>footer.cf-footer{display:block!important}</style>", unsafe_allow_html=True)
    runpy.run_path(str(Path(__file__).with_name("marketing.py")), run_name="__main__")
