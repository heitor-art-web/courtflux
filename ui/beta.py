from __future__ import annotations

import base64
import runpy
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BRAND_DIR = ROOT / "assets" / "brand"
APP_ICON = BRAND_DIR / "courtflux-app-icon.webp"
HORIZONTAL_LOGO = BRAND_DIR / "courtflux-horizontal-dark.webp"
PRODUCT_SPRITE = ROOT / "assets" / "site" / "courtflux-product-sprite.webp"

# Public marketing page and internal benchmark intentionally share one entrypoint.
# The existing benchmark remains isolated in beta_app.py and is not restructured.
_original_set_page_config = st.set_page_config
_original_title = st.title


def _page_icon():
    if not APP_ICON.exists():
        return "🏀"
    with Image.open(APP_ICON) as image:
        return image.copy()


def _repair_base64_encoded_webp(path: Path) -> None:
    """Repair a WebP that was accidentally committed as base64 text.

    The GitHub contents API used for the marketing asset encoded the binary once
    too many times, so the file on disk begins with the ASCII WebP base64 prefix
    ("UklG") instead of the binary RIFF header expected by Pillow. This keeps the
    app backwards-compatible with that commit while leaving valid binary WebPs
    untouched.
    """
    if not path.exists():
        return

    raw = path.read_bytes()
    if raw.startswith(b"RIFF"):
        return

    compact = b"".join(raw.split())
    if not compact.startswith(b"UklG"):
        return

    try:
        decoded = base64.b64decode(compact, validate=True)
    except Exception:
        return

    if decoded.startswith(b"RIFF") and b"WEBP" in decoded[:16]:
        path.write_bytes(decoded)


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
    # Repair only the known malformed marketing asset. A correctly committed WebP
    # is detected by its RIFF header and is never rewritten.
    _repair_base64_encoded_webp(PRODUCT_SPRITE)

    _original_set_page_config(
        page_title="CourtFlux — Basketball Intelligence from Video",
        page_icon=_page_icon(),
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    runpy.run_path(str(Path(__file__).with_name("marketing.py")), run_name="__main__")
