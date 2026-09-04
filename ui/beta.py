from __future__ import annotations

import base64
import runpy
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

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
    try:
        with Image.open(APP_ICON) as image:
            return image.copy()
    except (UnidentifiedImageError, OSError):
        return "🏀"


def _prepare_product_sprite(path: Path) -> None:
    """Best-effort repair/validation for the marketing sprite.

    Older checkouts may contain the sprite as base64 text instead of binary WebP.
    Decode that form when possible. If the file is still not a valid image, remove
    it for this checkout so the marketing page falls back gracefully instead of
    crashing during Pillow.Image.open().
    """
    if not path.exists():
        return

    raw = path.read_bytes()

    # Repair accidental base64-as-text versions.
    if not raw.startswith(b"RIFF"):
        compact = b"".join(raw.split())
        if compact.startswith(b"UklG"):
            try:
                decoded = base64.b64decode(compact, validate=True)
                if decoded.startswith(b"RIFF") and b"WEBP" in decoded[:16]:
                    path.write_bytes(decoded)
            except Exception:
                pass

    # Validate before marketing.py touches it. If invalid, make the asset absent;
    # marketing.py already has a safe visual fallback for missing assets.
    try:
        with Image.open(path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        try:
            path.unlink()
        except OSError:
            pass


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
    _prepare_product_sprite(PRODUCT_SPRITE)

    _original_set_page_config(
        page_title="CourtFlux — Basketball Intelligence from Video",
        page_icon=_page_icon(),
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    runpy.run_path(str(Path(__file__).with_name("marketing.py")), run_name="__main__")
