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
        :root{--bg:#050a10;--panel:#0b141e;--text:#f5f7fa;--muted:#98a7b6;--line:rgba(255,255,255,.09);--orange:#ff681f;--blue:#2a8cff;--green:#36d596}
        html{scroll-behavior:smooth}
        body,.stApp,[data-testid="stAppViewContainer"]{background:radial-gradient(circle at 80% 0,rgba(42,140,255,.09),transparent 34rem),var(--bg);color:var(--text)}
        [data-testid="stHeader"],#MainMenu,footer,[data-testid="stSidebar"]{display:none!important}
        .block-container{max-width:1360px;padding-top:1rem;padding-bottom:4rem}
        .cf-navlinks{display:flex;justify-content:flex-end;gap:1.2rem;align-items:center;min-height:74px}
        .cf-navlinks a{color:#aab6c2!important;text-decoration:none!important;font-size:.9rem;font-weight:600;white-space:nowrap}
        .cf-navlinks .cta{background:var(--orange);color:white!important;padding:.68rem .92rem;border-radius:9px}
        .cf-brand-img{display:block;width:100%;max-width:310px;height:auto;object-fit:contain}
        .cf-brand-img.footer{max-width:255px}
        .cf-section{padding:7rem 0 0;scroll-margin-top:80px}
        .cf-eyebrow,.cf-kicker{color:var(--orange);font-size:.73rem;letter-spacing:.14em;text-transform:uppercase;font-weight:800}
        .cf-title{font-size:clamp(2.55rem,5vw,5.7rem);line-height:.96;letter-spacing:-.055em;margin:.8rem 0 1.2rem;font-weight:760;max-width:900px}
        .cf-title span{color:var(--orange)}
        .cf-copy{color:var(--muted);font-size:clamp(1rem,1.4vw,1.18rem);line-height:1.72;max-width:760px}
        .cf-actions{display:flex;gap:.7rem;flex-wrap:wrap;margin-top:1.5rem}
        .cf-btn{display:inline-flex;min-height:45px;align-items:center;padding:0 1.05rem;border-radius:9px;text-decoration:none!important;font-weight:700;font-size:.9rem}
        .primary{background:var(--orange);color:white!important}.secondary{border:1px solid rgba(255,255,255,.16);color:#eef3f8!important;background:rgba(255,255,255,.025)}
        .cf-note{color:#738293;font-size:.8rem;margin-top:.85rem}
        div[data-testid="stImage"]{border:1px solid var(--line);border-radius:16px;overflow:hidden;background:#03080e;box-shadow:0 30px 70px rgba(0,0,0,.28)}
        div[data-testid="stImage"] img{display:block;width:100%!important;height:auto!important;object-fit:contain!important;object-position:center!important}
        .cf-product-fallback{min-height:320px;display:flex;align-items:center;justify-content:center;border:1px solid var(--line);border-radius:16px;color:#718191;background:#081018}
        .cf-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.8rem;margin-top:1.5rem}
        .cf-card{padding:1.25rem;border:1px solid var(--line);border-radius:13px;background:linear-gradient(180deg,rgba(13,23,34,.95),rgba(8,15,23,.94));min-height:178px}
        .cf-card h3{font-size:1.08rem;margin:.52rem 0 .42rem}.cf-card p{color:#8fa0af;line-height:1.58;font-size:.89rem;margin:0}
        .cf-badge{display:inline-block;margin-top:1rem;padding:.3rem .52rem;border:1px solid rgba(255,255,255,.1);border-radius:999px;color:#becad4;font-size:.66rem;text-transform:uppercase;letter-spacing:.06em}
        .cf-strip{display:grid;grid-template-columns:1.4fr repeat(4,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:12px;overflow:hidden;margin-top:2.5rem}
        .cf-strip div{background:#081018;padding:1rem;min-height:72px;display:flex;align-items:center;color:#d5dee6;font-size:.86rem;font-weight:650}.cf-strip div:first-child{color:#7f8e9d;text-transform:uppercase;font-size:.72rem;letter-spacing:.07em}
        .cf-lines{margin-top:1.3rem}.cf-line{padding:.72rem 0;border-bottom:1px solid rgba(255,255,255,.06);color:#c5d0da}.cf-line:before{content:'•';color:var(--orange);margin-right:.6rem}
        .cf-context{margin-top:1.1rem;padding:1rem 1.1rem;border:1px solid rgba(255,104,31,.2);border-radius:12px;background:rgba(255,104,31,.04)}
        .cf-context strong{color:#edf3f8}.cf-context p{margin:.35rem 0 0;color:#91a0af;line-height:1.55;font-size:.88rem}
        .cf-quote{border:1px solid var(--line);background:var(--panel);border-radius:14px;padding:1.6rem;font-size:clamp(1.3rem,2.3vw,2rem);line-height:1.35}.cf-quote span{color:var(--orange)}
        .cf-beta{margin-top:1.8rem;padding:clamp(1.5rem,4vw,3rem);border:1px solid rgba(255,104,31,.27);border-radius:17px;background:radial-gradient(circle at 88% 12%,rgba(255,104,31,.12),transparent 18rem),#09121b}
        .cf-status{display:grid;grid-template-columns:repeat(3,1fr);gap:.8rem;margin-top:1.2rem}.cf-status .cf-card{min-height:220px}.cf-list{list-style:none;padding:0;margin:.8rem 0 0}.cf-list li{padding:.3rem 0;color:#aebbc7;font-size:.86rem}.now{color:var(--green)}.doing{color:#f5a261}.later{color:#6e7f90}
        div[data-testid="stExpander"]{border:1px solid var(--line)!important;background:#081018!important;border-radius:11px!important;margin-bottom:.55rem!important}div[data-testid="stExpander"] summary{font-weight:650!important}div[data-testid="stExpander"] p{color:#9aa9b8!important;line-height:1.65!important}
        .cf-final{text-align:center;padding:8rem 0 4rem}.cf-final .cf-title,.cf-final .cf-copy{margin-left:auto;margin-right:auto}.cf-final .cf-actions{justify-content:center}
        .cf-footerlinks{display:flex;justify-content:flex-end;align-items:center;gap:1rem;min-height:82px}.cf-footerlinks a{color:#8898a8!important;text-decoration:none!important}.cf-footer-meta{color:#718191;font-size:.8rem;margin-top:.45rem}
        @media(max-width:900px){.cf-brand-img{max-width:230px}.cf-brand-img.footer{max-width:210px}.cf-navlinks a:not(.cta){display:none}.cf-grid,.cf-status{grid-template-columns:1fr}.cf-strip{grid-template-columns:1fr 1fr}.cf-strip div:first-child{grid-column:1/-1}.cf-section{padding-top:5.3rem}}
        </style>
        ''', unsafe_allow_html=True)

    logo_uri = file_data_uri(LOGO)
    nav_logo, nav_links = st.columns([.34, .66], vertical_alignment="center")
    with nav_logo:
        if logo_uri:
            st.markdown(f'<a href="#top" aria-label="CourtFlux"><img class="cf-brand-img" src="{logo_uri}" alt="CourtFlux — Basketball Intelligence"></a>', unsafe_allow_html=True)
        else:
            st.markdown("**COURTFLUX**")
    with nav_links:
        st.markdown('<div class="cf-navlinks"><a href="#product">Product</a><a href="#how">How it works</a><a href="#about">About</a><a href="#vision">Vision</a><a href="#faq">FAQ</a><a class="cta" href="?view=lab">Open Beta Lab</a></div>', unsafe_allow_html=True)
    st.markdown('<div id="top"></div>', unsafe_allow_html=True)

    left, right = st.columns([.9, 1.1], gap="large")
    with left:
        st.markdown('<div style="padding-top:4.4rem"><div class="cf-eyebrow">AI-powered basketball video analytics</div><h1 class="cf-title">Turn game film into <span>basketball intelligence.</span></h1><div class="cf-copy">CourtFlux tracks players and the ball, reconstructs movement on the court, and turns basketball video into structured information coaches and teams can act on.</div><div class="cf-actions"><a class="cf-btn primary" href="?view=lab">Open Beta Lab</a><a class="cf-btn secondary" href="#how">See how it works</a></div><div class="cf-note">Private beta · Built for coaches, academies, teams and performance analysts</div></div>', unsafe_allow_html=True)
    with right:
        show_visual(HERO_VISUAL, "CourtFlux full analyst workspace")

    st.markdown('<div class="cf-strip"><div>Designed for serious basketball development</div><div>Basketball Academies</div><div>Clubs & Teams</div><div>Coaches</div><div>Performance Analysts</div></div>', unsafe_allow_html=True)

    st.markdown('<section id="how" class="cf-section"><div class="cf-eyebrow">How it works</div><h2 class="cf-title">One video. Three layers of <span>understanding.</span></h2><div class="cf-copy">CourtFlux bridges raw game footage and the spatial, temporal and tactical information hidden inside each possession.</div></section>', unsafe_allow_html=True)
    a, b = st.columns([1.08, .92], gap="large")
    with a:
        show_visual(HOW_VISUAL, "Video to tracking to tactical insight")
    with b:
        st.markdown('<div class="cf-grid" style="grid-template-columns:1fr"><div class="cf-card"><div class="cf-kicker">01 · Video</div><h3>Use the game film you already have.</h3><p>Work from normal basketball footage rather than a proprietary capture system.</p></div><div class="cf-card"><div class="cf-kicker">02 · Tracking</div><h3>Computer vision follows the game.</h3><p>Players, teams and ball detections become persistent objects with temporal context.</p></div><div class="cf-card"><div class="cf-kicker">03 · Intelligence</div><h3>Movement becomes basketball information.</h3><p>Coordinates, possessions, events and tactical patterns become measurable and searchable.</p></div></div><div class="cf-context"><strong>Match context is progressive, not a blocker.</strong><p>Home and away teams are enough to start. Jersey numbers and player names are optional and can be added later when the user wants individual identity and roster-level analysis.</p></div>', unsafe_allow_html=True)

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

    st.markdown('<section id="about" class="cf-section"></section>', unsafe_allow_html=True)
    a, b = st.columns([1.02, .98], gap="large")
    with a:
        st.markdown('<div class="cf-eyebrow">About CourtFlux</div><h2 class="cf-title">Serious basketball technology, made <span>more accessible.</span></h2><div class="cf-copy">CourtFlux started from a simple question: why should advanced video analysis remain difficult or inaccessible for so many teams and academies?<br><br>We are building computer-vision and AI technology that can understand movement in basketball video and turn ordinary game film into useful information for coaching, development and performance.</div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="cf-quote">Statistics tell you <span>what</span> happened.<br>The video contains <span>why</span>.</div><div class="cf-copy" style="font-size:.95rem;margin-top:1rem">CourtFlux is designed to help coaches reach useful evidence faster — not to replace basketball judgement.</div>', unsafe_allow_html=True)

    st.markdown('<section id="vision" class="cf-section"><div class="cf-eyebrow">Our vision</div><h2 class="cf-title">Transform any recorded game into a source of <span>basketball intelligence.</span></h2><div class="cf-copy">The product grows in layers. Each layer should be useful on its own and create a stronger foundation for the next.</div><div class="cf-grid">' + card("01", "Video", "The game as it was recorded.") + card("02", "Tracking", "Persistent players, teams and ball state.") + card("03", "Spatial Data", "Objects mapped to court geometry.") + card("04", "Events", "Possessions, passes, shots and game actions.") + card("05", "Tactical Context", "Patterns understood in basketball terms.") + card("06", "Performance Intelligence", "Useful longitudinal insight for teams and players.") + '</div></section>', unsafe_allow_html=True)

    st.markdown('<section class="cf-section"><div class="cf-eyebrow">Values</div><h2 class="cf-title">The principles behind the <span>product.</span></h2><div class="cf-grid">' + card("01", "Precision over noise", "Show information because it is useful, not because the system can generate it.") + card("02", "Basketball first", "Technology exists to understand the game — not to demonstrate AI.") + card("03", "Evidence over opinion", "Insights should remain connected to observable video and data.") + card("04", "Coaches stay in control", "AI supports basketball judgement rather than trying to replace it.") + card("05", "Build for the real court", "Design for real cameras, occlusion, motion and imperfect footage.") + card("06", "Earn trust", "Be transparent about confidence, limitations and what is still being built.") + '</div></section>', unsafe_allow_html=True)

    st.markdown('<section class="cf-section"><div class="cf-eyebrow">Beta transparency</div><h2 class="cf-title">Build the foundation before the <span>promise.</span></h2><div class="cf-copy">The public product story separates what is working in the benchmark from active development and the longer-term coaching vision.</div><div class="cf-status"><div class="cf-card"><div class="cf-kicker">Available / benchmarked</div><ul class="cf-list"><li><span class="now">●</span> Player detection and tracking</li><li><span class="now">●</span> Team classification</li><li><span class="now">●</span> Ball detection</li><li><span class="now">●</span> Temporal ball tracking</li><li><span class="now">●</span> Processing diagnostics</li></ul></div><div class="cf-card"><div class="cf-kicker">In development</div><ul class="cf-list"><li><span class="doing">●</span> Player identity</li><li><span class="doing">●</span> Court mapping</li><li><span class="doing">●</span> Possession segmentation</li><li><span class="doing">●</span> Basketball event detection</li></ul></div><div class="cf-card"><div class="cf-kicker">Product vision</div><ul class="cf-list"><li><span class="later">●</span> Automated tactical insights</li><li><span class="later">●</span> Team pattern analysis</li><li><span class="later">●</span> Player development intelligence</li><li><span class="later">●</span> High-performance workflows</li></ul></div></div><div class="cf-beta"><div class="cf-eyebrow">CourtFlux Beta Lab</div><h2 class="cf-title" style="font-size:clamp(2rem,4vw,3.6rem)">The test environment stays a <span>test environment.</span></h2><div class="cf-copy">The existing benchmark interface remains unchanged. Configure clips, tune detector thresholds, run the pipeline, inspect diagnostic previews and compare continuity and performance metrics.</div><div class="cf-actions"><a class="cf-btn primary" href="?view=lab">Open the existing Beta Lab</a></div></div></section>', unsafe_allow_html=True)

    st.markdown('<section id="faq" class="cf-section"><div class="cf-eyebrow">FAQ</div><h2 class="cf-title">Questions about the <span>beta.</span></h2></section>', unsafe_allow_html=True)
    faq = [
        ("What kind of video can CourtFlux analyse?", "The current system is being developed around basketball game footage where the court and players are sufficiently visible. Camera position, resolution, occlusion and motion blur can affect quality."),
        ("Do I need specialist cameras?", "The product direction is to work with normal basketball video rather than require proprietary capture hardware."),
        ("How much match information do I need to provide?", "Home and away teams are enough to start. Jersey numbers and player names are optional and can be added later for player-level identity and roster-aware analysis."),
        ("Is CourtFlux a statistics tracker?", "No. Statistics can be an output, but the core proposition is video understanding: players, ball, movement, spatial context, possessions and eventually tactical interpretation."),
        ("Is the analysis real time?", "The current beta is designed around post-game or asynchronous processing. Real-time operation is not required for the initial product."),
        ("Can academies and youth clubs use it?", "Yes. Academies, youth clubs and coaching environments are central to the initial target market."),
        ("Does AI replace the coach or analyst?", "No. CourtFlux is an analysis assistant intended to surface evidence and reduce manual video work while keeping basketball judgement with the user."),
        ("What is available in the beta today?", "The benchmark focuses on detection, tracking, team classification, ball continuity, diagnostics and processing performance. Court mapping, possessions, events and tactical intelligence are being developed in stages."),
        ("Are uploaded videos private?", "Privacy and retention claims will match the final production architecture and published legal policy."),
    ]
    for question, answer in faq:
        with st.expander(question):
            st.write(answer)

    st.markdown('<div style="border-top:1px solid rgba(255,255,255,.07);margin-top:5rem;padding-top:2rem"></div>', unsafe_allow_html=True)
    st.markdown('<section class="cf-final"><div class="cf-eyebrow">CourtFlux</div><h2 class="cf-title">See more in the same <span>game film.</span></h2><div class="cf-copy">Turn basketball video into a structured foundation for tracking, analysis and better coaching decisions.</div><div class="cf-actions"><a class="cf-btn primary" href="?view=lab">Open Beta Lab</a><a class="cf-btn secondary" href="#top">Back to top</a></div></section>', unsafe_allow_html=True)

    footer_logo, footer_links = st.columns([.42, .58], vertical_alignment="bottom")
    with footer_logo:
        if logo_uri:
            st.markdown(f'<img class="cf-brand-img footer" src="{logo_uri}" alt="CourtFlux — Basketball Intelligence">', unsafe_allow_html=True)
        st.markdown('<div class="cf-footer-meta">© 2026 CourtFlux · Basketball Intelligence</div>', unsafe_allow_html=True)
    with footer_links:
        st.markdown('<div class="cf-footerlinks"><a href="#product">Product</a><a href="#about">About</a><a href="#vision">Vision</a><a href="#faq">FAQ</a><a href="?view=lab">Beta Lab</a></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    render_marketing_page()
