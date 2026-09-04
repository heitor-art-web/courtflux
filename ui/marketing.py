from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
BRAND_DIR = ROOT / "web" / "public" / "brand"

LOGO = BRAND_DIR / "horizontal_logo_safe_crop.png"
HERO_VISUAL = BRAND_DIR / "CourtFlux_Full_Analyst_Workspace.png"
HOW_VISUAL = BRAND_DIR / "CourtFlux_Video_Tracking_Tactical_Insight.png"
BALL_VISUAL = BRAND_DIR / "CourtFlux_Ball_Intelligence.png"
COURT_VISUAL = BRAND_DIR / "CourtFlux_Split_Screen_Court_Reconstruction.png"
COACH_VISUAL = BRAND_DIR / "CourtFlux_AI_Coach_Insight_Card.png"


def file_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/webp" if suffix == ".webp" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def show_visual(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.markdown(f'<div class="cf-product-fallback">{caption}</div>', unsafe_allow_html=True)


def card(kicker: str, title: str, body: str, status: str = "") -> str:
    badge = f'<div class="cf-badge">{status}</div>' if status else ""
    return f'<div class="cf-card"><div class="cf-kicker">{kicker}</div><h3>{title}</h3><p>{body}</p>{badge}</div>'


def render_marketing_page() -> None:
    st.markdown(
        '''
        <style>
        :root{--bg:#050a10;--panel:#0b141e;--text:#f5f7fa;--muted:#98a7b6;--line:rgba(255,255,255,.09);--orange:#ff681f;--orange2:#ff8a45;--blue:#2a8cff;--green:#36d596;--shadow:0 28px 80px rgba(0,0,0,.34)}
        html{scroll-behavior:smooth}
        body,.stApp,[data-testid="stAppViewContainer"]{background:radial-gradient(circle at 78% -5%,rgba(42,140,255,.12),transparent 32rem),radial-gradient(circle at 12% 14%,rgba(255,104,31,.07),transparent 28rem),linear-gradient(180deg,var(--bg),#04080d 72%);color:var(--text)}
        [data-testid="stHeader"],#MainMenu,footer,[data-testid="stSidebar"]{display:none!important}
        .block-container{max-width:1360px;padding-top:.65rem;padding-bottom:4rem}
        .cf-nav{position:sticky;top:.65rem;z-index:999;display:flex;align-items:center;justify-content:space-between;gap:1rem;min-height:72px;padding:.55rem .7rem .55rem .9rem;border:1px solid rgba(255,255,255,.08);border-radius:16px;background:rgba(5,10,16,.82);backdrop-filter:blur(18px);box-shadow:0 14px 38px rgba(0,0,0,.22)}
        .cf-nav-brand{display:flex;align-items:center;min-width:220px}.cf-nav-brand img{display:block;width:245px;max-width:32vw;height:auto}
        .cf-navlinks{display:flex;align-items:center;justify-content:flex-end;gap:1.05rem;flex-wrap:nowrap}.cf-navlinks a{color:#aab6c2!important;text-decoration:none!important;font-size:.87rem;font-weight:650;white-space:nowrap}.cf-navlinks a:hover{color:white!important}.cf-navlinks .login{color:#eef3f8!important}.cf-navlinks .cta{background:var(--orange);color:white!important;padding:.68rem .92rem;border-radius:10px;box-shadow:0 8px 24px rgba(255,104,31,.18)}.cf-navlinks .cta:hover{background:var(--orange2)}
        .cf-section{padding:7.2rem 0 0;scroll-margin-top:98px}.cf-eyebrow,.cf-kicker{color:var(--orange);font-size:.72rem;letter-spacing:.15em;text-transform:uppercase;font-weight:820}.cf-title{font-size:clamp(2.55rem,5vw,5.75rem);line-height:.95;letter-spacing:-.058em;margin:.8rem 0 1.15rem;font-weight:780;max-width:940px}.cf-title span{color:var(--orange)}.cf-copy{color:var(--muted);font-size:clamp(1rem,1.4vw,1.18rem);line-height:1.72;max-width:770px}
        .cf-actions{display:flex;gap:.72rem;flex-wrap:wrap;margin-top:1.55rem}.cf-btn{display:inline-flex;min-height:47px;align-items:center;justify-content:center;padding:0 1.12rem;border-radius:10px;text-decoration:none!important;font-weight:760;font-size:.9rem;transition:transform .18s ease,box-shadow .18s ease,background .18s ease}.cf-btn:hover{transform:translateY(-1px)}.primary{background:var(--orange);color:white!important;box-shadow:0 10px 28px rgba(255,104,31,.2)}.primary:hover{background:var(--orange2)}.secondary{border:1px solid rgba(255,255,255,.16);color:#eef3f8!important;background:rgba(255,255,255,.025)}.ghost{color:#9fb0bf!important;padding-left:.2rem;padding-right:.2rem}.cf-note{color:#738293;font-size:.81rem;margin-top:.92rem}
        .cf-hero-copy{padding-top:5.25rem;padding-bottom:1.7rem}.cf-hero-pills{display:flex;gap:.55rem;flex-wrap:wrap;margin-top:1.15rem}.cf-pill{border:1px solid rgba(255,255,255,.09);background:rgba(255,255,255,.025);color:#aebbc6;border-radius:999px;padding:.38rem .62rem;font-size:.72rem}
        div[data-testid="stImage"]{border:1px solid var(--line);border-radius:18px;overflow:hidden;background:#03080e;box-shadow:var(--shadow);transition:transform .28s ease,border-color .28s ease}div[data-testid="stImage"]:hover{transform:translateY(-2px);border-color:rgba(255,104,31,.28)}div[data-testid="stImage"] img{display:block;width:100%!important;height:auto!important;object-fit:contain!important;object-position:center!important}.cf-product-fallback{min-height:320px;display:flex;align-items:center;justify-content:center;border:1px solid var(--line);border-radius:16px;color:#718191;background:#081018}
        .cf-strip{display:grid;grid-template-columns:1.45fr repeat(4,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:14px;overflow:hidden;margin-top:2.7rem}.cf-strip div{background:#081018;padding:1rem;min-height:74px;display:flex;align-items:center;color:#d5dee6;font-size:.85rem;font-weight:680}.cf-strip div:first-child{color:#7f8e9d;text-transform:uppercase;font-size:.7rem;letter-spacing:.08em}
        .cf-signalbar{display:grid;grid-template-columns:repeat(4,1fr);gap:.7rem;margin-top:1rem}.cf-signal{padding:1rem 1.05rem;border:1px solid var(--line);border-radius:13px;background:linear-gradient(180deg,rgba(11,20,30,.98),rgba(7,14,22,.96))}.cf-signal strong{display:block;color:#f2f6f9;font-size:.92rem;margin-bottom:.22rem}.cf-signal span{color:#7f8f9f;font-size:.77rem;line-height:1.45}
        .cf-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.82rem;margin-top:1.55rem}.cf-card{padding:1.32rem;border:1px solid var(--line);border-radius:14px;background:linear-gradient(180deg,rgba(13,23,34,.96),rgba(8,15,23,.94));min-height:178px;transition:transform .2s ease,border-color .2s ease,background .2s ease}.cf-card:hover{transform:translateY(-3px);border-color:rgba(255,104,31,.22);background:linear-gradient(180deg,rgba(15,27,40,.98),rgba(8,15,23,.96))}.cf-card h3{font-size:1.08rem;margin:.52rem 0 .42rem}.cf-card p{color:#8fa0af;line-height:1.58;font-size:.89rem;margin:0}.cf-badge{display:inline-block;margin-top:1rem;padding:.3rem .54rem;border:1px solid rgba(255,255,255,.1);border-radius:999px;color:#becad4;font-size:.65rem;text-transform:uppercase;letter-spacing:.06em}
        .cf-lines{margin-top:1.3rem}.cf-line{padding:.74rem 0;border-bottom:1px solid rgba(255,255,255,.06);color:#c5d0da}.cf-line:before{content:'•';color:var(--orange);margin-right:.6rem}.cf-context{margin-top:1.1rem;padding:1.05rem 1.12rem;border:1px solid rgba(255,104,31,.2);border-radius:12px;background:rgba(255,104,31,.04)}.cf-context strong{color:#edf3f8}.cf-context p{margin:.35rem 0 0;color:#91a0af;line-height:1.55;font-size:.88rem}.cf-quote{border:1px solid var(--line);background:var(--panel);border-radius:15px;padding:1.7rem;font-size:clamp(1.3rem,2.3vw,2rem);line-height:1.35}.cf-quote span{color:var(--orange)}
        .cf-pricing{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;margin-top:1.7rem}.cf-price{position:relative;padding:1.35rem;border:1px solid var(--line);border-radius:17px;background:linear-gradient(180deg,#0b1621,#081018);min-height:470px;display:flex;flex-direction:column}.cf-price.featured{border-color:rgba(255,104,31,.44);box-shadow:0 24px 70px rgba(255,104,31,.09)}.cf-price-tag{position:absolute;top:1rem;right:1rem;padding:.32rem .52rem;border-radius:999px;background:rgba(255,104,31,.12);color:#ff9c64;font-size:.65rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase}.cf-price h3{font-size:1.25rem;margin:.42rem 0 .45rem}.cf-price .amount{font-size:2.25rem;letter-spacing:-.04em;font-weight:800;margin:.9rem 0 .2rem}.cf-price .amount small{font-size:.82rem;color:#7f8f9f;font-weight:650;letter-spacing:0}.cf-price p{color:#8fa0af;line-height:1.55;font-size:.86rem}.cf-price ul{list-style:none;padding:0;margin:1rem 0 1.25rem}.cf-price li{padding:.34rem 0;color:#c3ced7;font-size:.82rem}.cf-price li:before{content:'✓';color:var(--orange);margin-right:.5rem;font-weight:800}.cf-price .cf-btn{width:100%;margin-top:auto}.cf-usage{margin:.85rem 0 .35rem;padding:.72rem .8rem;border:1px solid rgba(42,140,255,.15);border-radius:10px;background:rgba(42,140,255,.045)}.cf-usage strong{display:block;color:#e8f0f7;font-size:.87rem}.cf-usage span{display:block;color:#7f91a1;font-size:.73rem;line-height:1.45;margin-top:.18rem}.cf-usage-explainer{display:grid;grid-template-columns:1.1fr .9fr;gap:.8rem;margin-top:1rem;padding:1.1rem 1.2rem;border:1px solid rgba(255,255,255,.08);border-radius:14px;background:#081018}.cf-usage-explainer strong{color:#edf3f8}.cf-usage-explainer p{margin:.3rem 0 0;color:#8fa0af;font-size:.84rem;line-height:1.55}.cf-usage-scale{display:flex;align-items:center;gap:.45rem;flex-wrap:wrap}.cf-usage-scale span{padding:.42rem .6rem;border:1px solid rgba(255,255,255,.08);border-radius:999px;color:#b8c5d0;font-size:.72rem}.cf-pricing-note{color:#728294;font-size:.78rem;margin-top:.8rem}
        .cf-account{display:grid;grid-template-columns:1.05fr .95fr;gap:.9rem;margin-top:1.6rem}.cf-account-card{padding:clamp(1.4rem,3vw,2rem);border:1px solid var(--line);border-radius:17px;background:radial-gradient(circle at 90% 8%,rgba(42,140,255,.09),transparent 18rem),#09131d}.cf-account-card.signup{border-color:rgba(255,104,31,.26);background:radial-gradient(circle at 88% 10%,rgba(255,104,31,.12),transparent 19rem),#0a131c}.cf-account-card h3{font-size:1.5rem;margin:.55rem 0 .55rem}.cf-account-card p{color:#8fa0af;line-height:1.62;max-width:620px}.cf-account-list{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:1rem}.cf-account-list span{font-size:.72rem;color:#aebbc7;border:1px solid rgba(255,255,255,.08);border-radius:999px;padding:.36rem .55rem}
        .cf-beta{margin-top:1.8rem;padding:clamp(1.5rem,4vw,3rem);border:1px solid rgba(255,104,31,.27);border-radius:17px;background:radial-gradient(circle at 88% 12%,rgba(255,104,31,.12),transparent 18rem),#09121b}.cf-status{display:grid;grid-template-columns:repeat(3,1fr);gap:.8rem;margin-top:1.2rem}.cf-status .cf-card{min-height:220px}.cf-list{list-style:none;padding:0;margin:.8rem 0 0}.cf-list li{padding:.3rem 0;color:#aebbc7;font-size:.86rem}.now{color:var(--green)}.doing{color:#f5a261}.later{color:#6e7f90}
        div[data-testid="stExpander"]{border:1px solid var(--line)!important;background:#081018!important;border-radius:11px!important;margin-bottom:.55rem!important}div[data-testid="stExpander"] summary{font-weight:650!important}div[data-testid="stExpander"] p{color:#9aa9b8!important;line-height:1.65!important}
        .cf-final{text-align:center;padding:8rem 0 4rem}.cf-final .cf-title,.cf-final .cf-copy{margin-left:auto;margin-right:auto}.cf-final .cf-actions{justify-content:center}.cf-footer{border-top:1px solid rgba(255,255,255,.07);padding-top:2rem}.cf-footer-grid{display:flex;justify-content:space-between;gap:1.5rem;align-items:flex-end}.cf-footer-brand img{display:block;width:255px;max-width:45vw;height:auto}.cf-footerlinks{display:flex;justify-content:flex-end;align-items:center;gap:1rem;flex-wrap:wrap}.cf-footerlinks a{color:#8898a8!important;text-decoration:none!important;font-size:.84rem}.cf-footer-meta{color:#718191;font-size:.78rem;margin-top:.5rem}
        @media(max-width:1100px){.cf-pricing{grid-template-columns:repeat(2,1fr)}.cf-navlinks a.hide-md{display:none}.cf-signalbar{grid-template-columns:1fr 1fr}}
        @media(max-width:900px){.cf-nav{position:relative;top:auto}.cf-nav-brand img{width:215px;max-width:48vw}.cf-navlinks{gap:.6rem}.cf-navlinks a.hide-mobile{display:none}.cf-strip{grid-template-columns:1fr 1fr}.cf-strip div:first-child{grid-column:1/-1}.cf-grid,.cf-status,.cf-account,.cf-usage-explainer{grid-template-columns:1fr}.cf-section{padding-top:5.5rem}.cf-hero-copy{padding-top:3.25rem}.cf-price{min-height:auto}.cf-footer-grid{align-items:flex-start;flex-direction:column}}
        @media(max-width:680px){.cf-pricing{grid-template-columns:1fr}}
        @media(max-width:560px){.block-container{padding-left:1rem;padding-right:1rem}.cf-nav{padding:.55rem}.cf-nav-brand{min-width:0}.cf-nav-brand img{width:190px}.cf-navlinks .login{display:none}.cf-navlinks .cta{padding:.62rem .72rem;font-size:.8rem}.cf-signalbar{grid-template-columns:1fr}}
        </style>
        ''',
        unsafe_allow_html=True,
    )

    logo_uri = file_data_uri(LOGO)
    logo_html = f'<img src="{logo_uri}" alt="CourtFlux — Basketball Intelligence">' if logo_uri else '<strong>COURTFLUX</strong>'
    st.markdown(
        f'''<nav class="cf-nav"><a class="cf-nav-brand" href="#top" aria-label="CourtFlux home">{logo_html}</a><div class="cf-navlinks"><a class="hide-mobile" href="#product">Product</a><a class="hide-md" href="#how">How it works</a><a href="#pricing">Pricing</a><a class="hide-md" href="?view=lab">Beta Lab</a><a class="login" href="?view=login">Sign in</a><a class="cta" href="?view=signup">Start free beta</a></div></nav><div id="top"></div>''',
        unsafe_allow_html=True,
    )

    left, right = st.columns([.9, 1.1], gap="large")
    with left:
        st.markdown(
            '''<div class="cf-hero-copy"><div class="cf-eyebrow">Basketball intelligence from ordinary game film</div><h1 class="cf-title">From game film to <span>decisions coaches can use.</span></h1><div class="cf-copy">Upload the footage you already have. CourtFlux follows players and the ball, builds spatial context, and turns movement into structured basketball intelligence.</div><div class="cf-actions"><a class="cf-btn primary" href="?view=signup">Start free beta</a><a class="cf-btn secondary" href="#how">See how it works</a><a class="cf-btn ghost" href="?view=lab">Open Beta Lab →</a></div><div class="cf-hero-pills"><span class="cf-pill">No proprietary camera required</span><span class="cf-pill">Coach-first AI</span><span class="cf-pill">Private beta</span></div></div>''',
            unsafe_allow_html=True,
        )
    with right:
        show_visual(HERO_VISUAL, "CourtFlux full analyst workspace")

    st.markdown(
        '''<div class="cf-strip"><div>Built for serious basketball development</div><div>Basketball Academies</div><div>Clubs & Teams</div><div>Coaches</div><div>Performance Analysts</div></div><div class="cf-signalbar"><div class="cf-signal"><strong>Use existing footage</strong><span>Start from normal game video.</span></div><div class="cf-signal"><strong>Track the game</strong><span>Players, teams and ball over time.</span></div><div class="cf-signal"><strong>Build spatial context</strong><span>Translate pixels into court movement.</span></div><div class="cf-signal"><strong>Reach useful evidence</strong><span>Surface moments coaches can review.</span></div></div>''',
        unsafe_allow_html=True,
    )

    st.markdown('<section id="how" class="cf-section"><div class="cf-eyebrow">How it works</div><h2 class="cf-title">One video. Three layers of <span>understanding.</span></h2><div class="cf-copy">CourtFlux bridges raw game footage and the spatial, temporal and tactical information hidden inside each possession.</div></section>', unsafe_allow_html=True)
    a, b = st.columns([1.08, .92], gap="large")
    with a:
        show_visual(HOW_VISUAL, "Video to tracking to tactical insight")
    with b:
        st.markdown('<div class="cf-grid" style="grid-template-columns:1fr"><div class="cf-card"><div class="cf-kicker">01 · Video</div><h3>Use the game film you already have.</h3><p>Work from normal basketball footage rather than a proprietary capture system.</p></div><div class="cf-card"><div class="cf-kicker">02 · Tracking</div><h3>Computer vision follows the game.</h3><p>Players, teams and ball detections become persistent objects with temporal context.</p></div><div class="cf-card"><div class="cf-kicker">03 · Intelligence</div><h3>Movement becomes basketball information.</h3><p>Coordinates, possessions, events and tactical patterns become measurable and searchable.</p></div></div><div class="cf-context"><strong>Match context is progressive, not a blocker.</strong><p>Home and away teams are enough to start. Jersey numbers and player names can be added later when roster-level identity matters.</p></div>', unsafe_allow_html=True)

    st.markdown('<section id="product" class="cf-section"></section>', unsafe_allow_html=True)
    a, b = st.columns([.84, 1.16], gap="large")
    with a:
        st.markdown('<div class="cf-eyebrow">Ball Intelligence</div><h2 class="cf-title">Follow the object that defines the <span>possession.</span></h2><div class="cf-copy">Ball quality is a foundational dependency for possession, passing, shot, rebound and tactical analysis. CourtFlux combines direct detections with temporal continuity to provide a more reliable ball state.</div><div class="cf-lines"><div class="cf-line"><b>Direct detection</b> — frame-level ball candidates</div><div class="cf-line"><b>Temporal tracking</b> — continuity through short gaps</div><div class="cf-line"><b>Reacquisition</b> — recovery when the ball becomes visible again</div></div>', unsafe_allow_html=True)
    with b:
        show_visual(BALL_VISUAL, "CourtFlux Ball Intelligence")

    st.markdown('<div class="cf-section">', unsafe_allow_html=True)
    a, b = st.columns([1.12, .88], gap="large")
    with a:
        show_visual(COURT_VISUAL, "CourtFlux split-screen court reconstruction")
    with b:
        st.markdown('<div class="cf-eyebrow">Court Reconstruction</div><h2 class="cf-title">From broadcast pixels to <span>spatial intelligence.</span></h2><div class="cf-copy">Map tracked objects from video coordinates onto a basketball court so coaches can reason about spacing, coverage, trajectories and team shape instead of isolated boxes.</div><div class="cf-lines"><div class="cf-line">Player positions and trajectories</div><div class="cf-line">Ball position and movement</div><div class="cf-line">Offensive spacing and defensive structure</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="cf-section">', unsafe_allow_html=True)
    a, b = st.columns([.86, 1.14], gap="large")
    with a:
        st.markdown('<div class="cf-eyebrow">Tactical Intelligence · Vision</div><h2 class="cf-title">Not just what happened. <span>Why it happened.</span></h2><div class="cf-copy">The long-term CourtFlux vision is an AI assistant for coaches that connects tracked movement and events to interpretable tactical context: late rotations, weak-side help, paint penetration, spacing breakdowns and open-corner creation.</div><div class="cf-note">Product direction — not every tactical insight is available in the current beta.</div>', unsafe_allow_html=True)
    with b:
        show_visual(COACH_VISUAL, "CourtFlux AI coach insight concept")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<section class="cf-section"><div class="cf-eyebrow">Product capabilities</div><h2 class="cf-title">Built around the <span>game itself.</span></h2><div class="cf-copy">CourtFlux is not a traditional box-score tracker. The platform is built around objects, movement, possessions and spatial context extracted from video.</div><div class="cf-grid">' + card("Computer Vision", "Player Tracking", "Detect and persist players across game footage.", "Beta") + card("Core Priority", "Ball Intelligence", "Detect, associate and maintain the ball state across time.", "Beta") + card("Classification", "Team Identification", "Separate teams consistently from visual appearance.", "Beta") + card("Spatial Layer", "Court Mapping", "Translate image positions into meaningful court coordinates.", "In development") + card("Game Structure", "Possession Analysis", "Segment continuous footage into basketball possessions.", "In development") + card("Coaching Layer", "Tactical Intelligence", "Explain patterns and moments in coaching language.", "Vision") + '</div></section>', unsafe_allow_html=True)

    st.markdown('<section class="cf-section"><div class="cf-eyebrow">Who it is for</div><h2 class="cf-title">For people who need to <span>understand the game.</span></h2><div class="cf-grid">' + card("01", "Basketball Academies", "Create a repeatable video-analysis workflow across teams and age groups.") + card("02", "Clubs & Teams", "Turn game film into a structured layer of performance information.") + card("03", "Coaches", "Spend less time searching video and more time reviewing relevant moments.") + card("04", "Performance Analysts", "Connect video, coordinates, trajectories and events in one workflow.") + card("05", "Development Staff", "Build longitudinal understanding around movement and decision-making.") + card("06", "Future Player Access", "Extend selected intelligence to players and families over time.") + '</div></section>', unsafe_allow_html=True)

    st.markdown(
        '''<section id="pricing" class="cf-section"><div class="cf-eyebrow">Pricing</div><h2 class="cf-title">Pay for the video you <span>actually analyse.</span></h2><div class="cf-copy">CourtFlux plans are designed around analysis minutes rather than a fixed number of games. A short clip only uses its own duration, while a full-game recording is simply a larger share of the same monthly allowance.</div><div class="cf-pricing"><div class="cf-price"><div class="cf-kicker">Private beta</div><h3>Beta</h3><div class="amount">€0 <small>/ month</small></div><p>Explore the core tracking layer while CourtFlux is being benchmarked and refined.</p><div class="cf-usage"><strong>120 analysis minutes / month</strong><span>About 2 full-game recordings, or many shorter clips.</span></div><ul><li>Player tracking</li><li>Team classification</li><li>Ball intelligence</li><li>Basic analysis and diagnostics</li></ul><a class="cf-btn secondary" href="?view=signup">Start free beta</a></div><div class="cf-price"><div class="cf-kicker">Individual workflow</div><h3>Coach</h3><div class="amount">€14.99 <small>/ month</small></div><p>For an individual coach or analyst who wants useful basketball moments without hours of manual review.</p><div class="cf-usage"><strong>300 analysis minutes / month</strong><span>About 5 full-game recordings, or dozens of shorter clips.</span></div><ul><li>Everything in Beta</li><li>Possession and event primitives as they launch</li><li>Saved analyses and game summaries</li><li>One coach workspace</li></ul><a class="cf-btn secondary" href="?view=signup">Join the beta</a></div><div class="cf-price featured"><div class="cf-price-tag">Best for teams</div><div class="cf-kicker">Shared team workflow</div><h3>Team</h3><div class="amount">€29.99 <small>/ month</small></div><p>For one squad and coaching staff building a repeatable season-long analysis workflow.</p><div class="cf-usage"><strong>720 analysis minutes / month</strong><span>About 12 full-game recordings, with clips charged only by duration.</span></div><ul><li>Everything in Coach</li><li>Saved roster and player identity</li><li>Structured player and team analytics</li><li>Shared staff access and exports</li></ul><a class="cf-btn primary" href="?view=signup">Join the beta</a></div><div class="cf-price"><div class="cf-kicker">Multi-team development</div><h3>Academy</h3><div class="amount">From €79 <small>/ month</small></div><p>For academies and clubs managing multiple squads, staff members and player development pathways.</p><div class="cf-usage"><strong>2,400 analysis minutes / month</strong><span>About 40 full-game recordings across the organisation.</span></div><ul><li>Up to 5 teams</li><li>Multiple rosters and staff permissions</li><li>Player development history and cross-team reporting</li><li>Tactical intelligence as it becomes available</li></ul><a class="cf-btn secondary" href="?view=signup">Talk to CourtFlux</a></div></div><div class="cf-usage-explainer"><div><strong>What counts as a game?</strong><p>A game is a basketball session in your library, usually tied to one matchup and date. Billing is not based on the number of game records — it is based on processed video duration.</p></div><div><div class="cf-usage-scale"><span>Short clip · up to 5 min</span><span>Segment · 5–20 min</span><span>Full game · up to 60 min</span></div><p>Short clips only consume the minutes they use. Footage over 60 minutes simply continues against the same analysis-minute allowance.</p></div></div><div class="cf-pricing-note">Indicative product pricing only. Billing is not active during private beta. Final allowances, feature availability and paid launch terms will be confirmed before charging customers.</div></section>''',
        unsafe_allow_html=True,
    )

    st.markdown(
        '''<section id="account" class="cf-section"><div class="cf-eyebrow">Account access</div><h2 class="cf-title">A clear path from <span>interest to analysis.</span></h2><div class="cf-copy">New users should be able to understand the value first, then create an account when they are ready. Existing beta users get a direct path back into CourtFlux without hunting through the site.</div><div class="cf-account"><div class="cf-account-card signup"><div class="cf-kicker">New to CourtFlux</div><h3>Create your account</h3><p>Join the private beta, tell us what kind of basketball environment you work in, and start with the game film you already have.</p><div class="cf-account-list"><span>Coach</span><span>Team</span><span>Academy</span><span>Analyst</span></div><div class="cf-actions"><a class="cf-btn primary" href="?view=signup">Start free beta</a></div></div><div class="cf-account-card"><div class="cf-kicker">Already invited</div><h3>Sign in and continue</h3><p>Return to your CourtFlux workspace, saved context and analysis flow. Beta Lab access remains available separately for technical testing.</p><div class="cf-actions"><a class="cf-btn secondary" href="?view=login">Sign in</a><a class="cf-btn ghost" href="?view=lab">Open Beta Lab →</a></div></div></div></section>''',
        unsafe_allow_html=True,
    )

    st.markdown('<section id="about" class="cf-section"></section>', unsafe_allow_html=True)
    a, b = st.columns([1.02, .98], gap="large")
    with a:
        st.markdown('<div class="cf-eyebrow">About CourtFlux</div><h2 class="cf-title">Serious basketball technology, made <span>more accessible.</span></h2><div class="cf-copy">CourtFlux started from a simple question: why should advanced video analysis remain difficult or inaccessible for so many teams and academies?<br><br>We are building computer-vision and AI technology that can understand movement in basketball video and turn ordinary game film into useful information for coaching, development and performance.</div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="cf-quote">Statistics tell you <span>what</span> happened.<br>The video contains <span>why</span>.</div><div class="cf-copy" style="font-size:.95rem;margin-top:1rem">CourtFlux is designed to help coaches reach useful evidence faster — not to replace basketball judgement.</div>', unsafe_allow_html=True)

    st.markdown('<section id="vision" class="cf-section"><div class="cf-eyebrow">Our vision</div><h2 class="cf-title">Transform any recorded game into a source of <span>basketball intelligence.</span></h2><div class="cf-copy">The product grows in layers. Each layer should be useful on its own and create a stronger foundation for the next.</div><div class="cf-grid">' + card("01", "Video", "The game as it was recorded.") + card("02", "Tracking", "Persistent players, teams and ball state.") + card("03", "Spatial Data", "Objects mapped to court geometry.") + card("04", "Events", "Possessions, passes, shots and game actions.") + card("05", "Tactical Context", "Patterns understood in basketball terms.") + card("06", "Performance Intelligence", "Useful longitudinal insight for teams and players.") + '</div></section>', unsafe_allow_html=True)

    st.markdown('<section class="cf-section"><div class="cf-eyebrow">Values</div><h2 class="cf-title">The principles behind the <span>product.</span></h2><div class="cf-grid">' + card("01", "Precision over noise", "Show information because it is useful, not because the system can generate it.") + card("02", "Basketball first", "Technology exists to understand the game — not to demonstrate AI.") + card("03", "Evidence over opinion", "Insights should remain connected to observable video and data.") + card("04", "Coaches stay in control", "AI supports basketball judgement rather than trying to replace it.") + card("05", "Build for the real court", "Design for real cameras, occlusion, motion and imperfect footage.") + card("06", "Earn trust", "Be transparent about confidence, limitations and what is still being built.") + '</div></section>', unsafe_allow_html=True)

    st.markdown('<section class="cf-section"><div class="cf-eyebrow">Beta transparency</div><h2 class="cf-title">Build the foundation before the <span>promise.</span></h2><div class="cf-copy">The public product story separates what is working in the benchmark from active development and the longer-term coaching vision.</div><div class="cf-status"><div class="cf-card"><div class="cf-kicker">Available / benchmarked</div><ul class="cf-list"><li><span class="now">●</span> Player detection and tracking</li><li><span class="now">●</span> Team classification</li><li><span class="now">●</span> Ball detection</li><li><span class="now">●</span> Temporal ball tracking</li><li><span class="now">●</span> Processing diagnostics</li></ul></div><div class="cf-card"><div class="cf-kicker">In development</div><ul class="cf-list"><li><span class="doing">●</span> Player identity</li><li><span class="doing">●</span> Court mapping</li><li><span class="doing">●</span> Possession segmentation</li><li><span class="doing">●</span> Basketball event detection</li></ul></div><div class="cf-card"><div class="cf-kicker">Product vision</div><ul class="cf-list"><li><span class="later">●</span> Automated tactical insights</li><li><span class="later">●</span> Team pattern analysis</li><li><span class="later">●</span> Player development intelligence</li><li><span class="later">●</span> High-performance workflows</li></ul></div></div><div class="cf-beta"><div class="cf-eyebrow">CourtFlux Beta Lab</div><h2 class="cf-title" style="font-size:clamp(2rem,4vw,3.6rem)">The test environment stays a <span>test environment.</span></h2><div class="cf-copy">Configure clips, tune detector thresholds, run the pipeline, inspect diagnostic previews and compare continuity and performance metrics without mixing engineering controls into the customer experience.</div><div class="cf-actions"><a class="cf-btn secondary" href="?view=lab">Open Beta Lab</a></div></div></section>', unsafe_allow_html=True)

    st.markdown('<section id="faq" class="cf-section"><div class="cf-eyebrow">FAQ</div><h2 class="cf-title">Questions about the <span>beta.</span></h2></section>', unsafe_allow_html=True)
    faq = [
        ("What kind of video can CourtFlux analyse?", "The current system is being developed around basketball game footage where the court and players are sufficiently visible. Camera position, resolution, occlusion and motion blur can affect quality."),
        ("Do I need specialist cameras?", "The product direction is to work with normal basketball video rather than require proprietary capture hardware."),
        ("How much match information do I need to provide?", "Home and away teams are enough to start. Jersey numbers and player names are optional and can be added later for player-level identity and roster-aware analysis."),
        ("How do analysis minutes work?", "Usage is based on processed video duration, not the number of uploads or game records. A 3-minute clip uses about 3 analysis minutes; a 60-minute recording uses about 60. This keeps short clips economical while making full-game usage predictable."),
        ("What counts as a full game?", "For pricing guidance, a full game means up to 60 minutes of uploaded video. A game in the product is still a normal matchup/session record, but billing is based on the actual video duration processed."),
        ("Is CourtFlux a statistics tracker?", "No. Statistics can be an output, but the core proposition is video understanding: players, ball, movement, spatial context, possessions and eventually tactical interpretation."),
        ("Is the analysis real time?", "The current beta is designed around post-game or asynchronous processing. Real-time operation is not required for the initial product."),
        ("Can academies and youth clubs use it?", "Yes. Academies, youth clubs and coaching environments are central to the initial target market."),
        ("Does AI replace the coach or analyst?", "No. CourtFlux is an analysis assistant intended to surface evidence and reduce manual video work while keeping basketball judgement with the user."),
        ("What does the beta cost?", "Private beta access is currently presented as free while the product is being benchmarked. Planned paid tiers and analysis-minute allowances are indicative until billing and final feature packaging are ready."),
        ("What is available in the beta today?", "The benchmark focuses on detection, tracking, team classification, ball continuity, diagnostics and processing performance. Court mapping, possessions, events and tactical intelligence are being developed in stages."),
        ("Are uploaded videos private?", "Privacy and retention claims will match the final production architecture and published legal policy."),
    ]
    for question, answer in faq:
        with st.expander(question):
            st.write(answer)

    st.markdown('<section class="cf-final"><div class="cf-eyebrow">CourtFlux</div><h2 class="cf-title">See more in the same <span>game film.</span></h2><div class="cf-copy">Turn basketball video into a structured foundation for tracking, analysis and better coaching decisions.</div><div class="cf-actions"><a class="cf-btn primary" href="?view=signup">Start free beta</a><a class="cf-btn secondary" href="#pricing">See pricing</a></div></section>', unsafe_allow_html=True)

    st.markdown(f'<footer class="cf-footer"><div class="cf-footer-grid"><div class="cf-footer-brand">{logo_html}<div class="cf-footer-meta">© 2026 CourtFlux · Basketball Intelligence</div></div><div class="cf-footerlinks"><a href="#product">Product</a><a href="#pricing">Pricing</a><a href="#about">About</a><a href="#vision">Vision</a><a href="#faq">FAQ</a><a href="?view=login">Sign in</a><a href="?view=lab">Beta Lab</a></div></div></footer>', unsafe_allow_html=True)


if __name__ == "__main__":
    render_marketing_page()
