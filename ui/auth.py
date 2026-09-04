from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
BRAND_DIR = ROOT / "web" / "public" / "brand"
LOGO = BRAND_DIR / "horizontal_logo_safe_crop.png"


def _data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _mode() -> str:
    try:
        value = st.query_params.get("view", "login")
        if isinstance(value, list):
            value = value[0] if value else "login"
        return "signup" if str(value).lower() == "signup" else "login"
    except Exception:
        return "login"


def render_auth_page() -> None:
    mode = _mode()
    logo_uri = _data_uri(LOGO)
    logo_html = f'<img src="{logo_uri}" alt="CourtFlux">' if logo_uri else '<strong>COURTFLUX</strong>'

    st.markdown(
        """
        <style>
        :root{--bg:#050a10;--panel:#0b141e;--text:#f5f7fa;--muted:#98a7b6;--line:rgba(255,255,255,.09);--orange:#ff681f;--blue:#2a8cff}
        html,body,.stApp,[data-testid="stAppViewContainer"]{background:radial-gradient(circle at 78% 0,rgba(42,140,255,.11),transparent 31rem),radial-gradient(circle at 12% 16%,rgba(255,104,31,.08),transparent 26rem),#050a10;color:var(--text)}
        [data-testid="stHeader"],#MainMenu,footer,[data-testid="stSidebar"]{display:none!important}
        .block-container{max-width:1080px;padding-top:2rem;padding-bottom:4rem}
        .cf-auth-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:3.2rem}
        .cf-auth-top img{display:block;width:245px;max-width:48vw;height:auto}
        .cf-auth-top a{color:#aab6c2!important;text-decoration:none!important;font-size:.88rem;font-weight:650}
        .cf-auth-shell{display:grid;grid-template-columns:.9fr 1.1fr;gap:1rem;align-items:stretch}
        .cf-auth-story,.cf-auth-card{border:1px solid var(--line);border-radius:18px;background:linear-gradient(180deg,rgba(12,23,34,.97),rgba(7,14,22,.97));box-shadow:0 30px 80px rgba(0,0,0,.28)}
        .cf-auth-story{padding:clamp(1.5rem,4vw,3rem);background:radial-gradient(circle at 82% 12%,rgba(255,104,31,.13),transparent 18rem),#09131d;display:flex;flex-direction:column;justify-content:space-between;min-height:540px}
        .cf-auth-card{padding:clamp(1.3rem,3vw,2rem)}
        .cf-kicker{color:var(--orange);font-size:.72rem;letter-spacing:.15em;text-transform:uppercase;font-weight:820}
        .cf-auth-story h1{font-size:clamp(2.2rem,4.5vw,4.7rem);line-height:.96;letter-spacing:-.055em;margin:.8rem 0 1.1rem}
        .cf-auth-story h1 span{color:var(--orange)}
        .cf-auth-story p{color:#98a7b6;line-height:1.68;max-width:540px}
        .cf-points{display:grid;gap:.65rem;margin-top:1.5rem}.cf-point{padding:.9rem 1rem;border:1px solid rgba(255,255,255,.07);border-radius:12px;background:rgba(255,255,255,.02)}
        .cf-point strong{display:block;font-size:.9rem}.cf-point span{display:block;color:#7f8f9e;font-size:.8rem;margin-top:.2rem;line-height:1.45}
        .cf-auth-card h2{font-size:1.8rem;margin:.45rem 0 .45rem}.cf-auth-card p{color:#8fa0af;line-height:1.55;margin-top:0}
        div[data-testid="stTextInput"] input,div[data-testid="stSelectbox"]>div>div{background:#071019!important;border-color:rgba(255,255,255,.1)!important}
        div[data-testid="stTextInput"] label,div[data-testid="stSelectbox"] label{color:#c9d3dc!important;font-size:.82rem!important}
        div[data-testid="stFormSubmitButton"] button{width:100%;min-height:46px;border-radius:10px;background:#ff681f;border:0;color:white;font-weight:760}
        .cf-switch{margin-top:1rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,.07);font-size:.84rem;color:#8292a1}.cf-switch a{color:#ff8f51!important;text-decoration:none!important;font-weight:700}
        .cf-auth-note{margin-top:1rem;padding:.85rem 1rem;border:1px solid rgba(42,140,255,.18);border-radius:11px;background:rgba(42,140,255,.05);color:#90a4b7;font-size:.78rem;line-height:1.55}
        @media(max-width:820px){.cf-auth-shell{grid-template-columns:1fr}.cf-auth-story{min-height:auto}.block-container{padding-left:1rem;padding-right:1rem}}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="cf-auth-top"><a href="?view=site" aria-label="CourtFlux home">{logo_html}</a><a href="?view=site">← Back to site</a></div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([.9, 1.1], gap="large")
    with left:
        if mode == "signup":
            st.markdown(
                """
                <div class="cf-auth-story">
                  <div><div class="cf-kicker">Private beta</div><h1>Bring your game film. <span>We build the intelligence.</span></h1><p>Create a CourtFlux account to join the beta and help shape a basketball-first analysis workflow around the footage teams already record.</p></div>
                  <div class="cf-points"><div class="cf-point"><strong>Start with team context</strong><span>Home and away teams are enough. Player identity can come later.</span></div><div class="cf-point"><strong>Keep the workflow coach-first</strong><span>Evidence and video stay connected to basketball decisions.</span></div><div class="cf-point"><strong>Grow with the product</strong><span>Tracking first, then spatial context, events and tactical intelligence.</span></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="cf-auth-story">
                  <div><div class="cf-kicker">Welcome back</div><h1>Get back to the <span>game.</span></h1><p>Sign in to continue your CourtFlux workflow. The customer experience stays separate from the technical Beta Lab used for benchmarking.</p></div>
                  <div class="cf-points"><div class="cf-point"><strong>One basketball workspace</strong><span>Teams, game context and analysis should live together.</span></div><div class="cf-point"><strong>Fast return path</strong><span>No need to navigate through marketing content once you already have access.</span></div><div class="cf-point"><strong>Beta Lab remains available</strong><span>Technical testing stays isolated from the product experience.</span></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        if mode == "signup":
            st.markdown('<div class="cf-kicker">Create account</div><h2>Join CourtFlux</h2><p>Tell us who you are so the onboarding flow can fit your basketball environment.</p>', unsafe_allow_html=True)
            with st.form("courtflux_signup", clear_on_submit=False):
                name = st.text_input("Full name")
                email = st.text_input("Email")
                organization = st.text_input("Team / academy / organization")
                role = st.selectbox("Role", ["Coach", "Performance analyst", "Academy / club staff", "Team staff", "Other"])
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Create account")
            if submitted:
                st.info("The sign-up UI is ready. Account persistence and authentication still need to be connected to the production auth backend before this can create a real user.")
            st.markdown('<div class="cf-switch">Already have beta access? <a href="?view=login">Sign in</a></div><div class="cf-auth-note">Front-end authentication flow only: credentials are not persisted by CourtFlux yet. The production auth service should be connected before public launch.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="cf-kicker">Account access</div><h2>Sign in</h2><p>Use your CourtFlux account to continue where you left off.</p>', unsafe_allow_html=True)
            with st.form("courtflux_login", clear_on_submit=False):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign in")
            if submitted:
                st.info("The sign-in UI is ready. Authentication still needs to be connected to the production auth backend before login can create a real session.")
            st.markdown('<div class="cf-switch">Need an account? <a href="?view=signup">Join the private beta</a> · <a href="?view=lab">Open Beta Lab</a></div><div class="cf-auth-note">Front-end authentication flow only: credentials are not persisted by CourtFlux yet. The production auth service should be connected before public launch.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    render_auth_page()
