from __future__ import annotations

import runpy
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
BRAND_DIR = ROOT / "web" / "public" / "brand"
APP_ICON = BRAND_DIR / "courtflux_app_icon_128px.png"
HORIZONTAL_LOGO = BRAND_DIR / "horizontal_logo_safe_crop.png"

# Streamlit is now the engineering/private-beta analysis surface only.
# Public marketing, authentication and account/dashboard flows live in the
# canonical Next.js application under web/.
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
    kwargs["page_title"] = "CourtFlux Beta Lab"
    kwargs["page_icon"] = _page_icon()
    return _original_set_page_config(*args, **kwargs)


def _courtflux_title(body, *args, **kwargs):
    # Keep the existing benchmark controls/layout intact; only replace the title
    # with the approved CourtFlux horizontal lockup from web/public/brand.
    if "CourtFlux" in str(body) and HORIZONTAL_LOGO.exists():
        st.image(str(HORIZONTAL_LOGO), width=560)
        return None
    return _original_title(body, *args, **kwargs)


st.set_page_config = _courtflux_page_config
st.title = _courtflux_title
runpy.run_path(str(Path(__file__).with_name("beta_app.py")), run_name="__main__")
