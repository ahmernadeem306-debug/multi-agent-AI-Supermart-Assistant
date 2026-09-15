"""BizAgent visual theme — "Corner Grocer" identity.

Pure presentation layer: fonts, colours, a hero banner and styled page
headers for the Streamlit UI. Nothing here touches the API, agents,
database or any other backend logic — it only injects CSS/HTML and
renders text that pages already had.
"""
from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Palette — a warm produce-market identity: deep grocer's-awning green,
# paper-crate cream, ripe-tomato accent, wheat gold for price-tag details.
# ---------------------------------------------------------------------------
FOREST = "#1F4D3D"
FOREST_DARK = "#163A2D"
LEAF = "#4C8C6B"
CREAM = "#FAF6EC"
PAPER = "#FFFFFF"
INK = "#25312B"
PAPRIKA = "#D9552C"
PAPRIKA_DARK = "#B8431F"
WHEAT = "#E7B65B"
LINE = "#E3DCC8"

# A small tiled line-art pattern (leaves + a crate grid) used behind the
# hero banner, drawn as an inline SVG data-URI — original vector artwork,
# not a photograph, so it renders instantly with no external dependency.
_PATTERN_SVG = (
    "data:image/svg+xml;utf8,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E"
    "%3Cg fill='none' stroke='%23FAF6EC' stroke-width='1.4' opacity='0.16'%3E"
    "%3Ccircle cx='20' cy='20' r='11'/%3E"
    "%3Cpath d='M20 9c4 4 4 14 0 22'/%3E"
    "%3Crect x='64' y='64' width='36' height='28' rx='2'/%3E"
    "%3Cpath d='M64 72h36M64 80h36M72 64v28M88 64v28'/%3E"
    "%3Cpath d='M96 18c-8 0-13 6-13 13 7 3 15-1 13-13z'/%3E"
    "%3Ccircle cx='16' cy='96' r='7'/%3E"
    "%3Ccircle cx='34' cy='100' r='5'/%3E"
    "%3C/g%3E%3C/svg%3E"
)


# ---------------------------------------------------------------------------
# A small original line-icon set (hand-drawn paths, Feather-style stroke
# icons) so the app uses consistent vector icons instead of emoji, which
# render differently across operating systems and browsers.
# ---------------------------------------------------------------------------
_ICON_PATHS: dict[str, str] = {
    "cart": (
        '<path d="M3 4h2.2l1 12.2A2 2 0 0 0 8.2 18h9.1a2 2 0 0 0 2-1.7L20.5 8H6.4"/>'
        '<circle cx="9.5" cy="21.5" r="1.4"/><circle cx="17" cy="21.5" r="1.4"/>'
    ),
    "chart": (
        '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>'
    ),
    "chat": (
        '<path d="M4 5h16v11H9l-4 4v-4H4z"/><path d="M8 9.5h8M8 12.5h5"/>'
    ),
    "search": (
        '<circle cx="10.5" cy="10.5" r="6.5"/><path d="M20 20l-4.7-4.7"/>'
    ),
    "trend": (
        '<path d="M3 17l6-6 4 4 8-9"/><path d="M15 6h6v6"/>'
    ),
    "book": (
        '<path d="M4 5.5C6 4.5 9 4 12 5.5c3-1.5 6-1 8 0v13c-2-1-5-1.5-8 0-3-1.5-6-1-8 0z"/>'
        '<path d="M12 5.5v13"/>'
    ),
    "receipt": (
        '<path d="M6 3h12v18l-2.5-1.6L13 21l-2.5-1.6L8 21l-2-1.6z"/>'
        '<path d="M9 8h6M9 11.5h6M9 15h4"/>'
    ),
    "leaf": (
        '<path d="M5 19C4 11 10 4 20 4c1 8-6 15-15 15z"/><path d="M5 19c2-5 5-8 10-10"/>'
    ),
}


def icon_svg(name: str, size: int = 22, color: str = "currentColor") -> str:
    """Return an inline SVG for one of the app's original line icons."""
    body = _ICON_PATHS.get(name, _ICON_PATHS["leaf"])
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round" style="vertical-align:middle">{body}</svg>'
    )


def render_html(html: str) -> None:
    """st.markdown(..., unsafe_allow_html=True), safe against Markdown quirks.

    Two Markdown behaviours fight raw HTML/CSS injection here:
    1. A line indented 4+ spaces becomes a preformatted code block.
    2. A block that starts with a tag like <link> or <h1> (both are on
       Markdown's list of block-level tags) is parsed as "raw HTML" only
       until the *first blank line* — even if that's in the middle of an
       open <style> tag. Our CSS had blank lines between rule groups for
       readability, so Markdown cut the raw-HTML block off partway through
       and rendered the rest of the CSS as plain paragraph text (the exact
       bug the screenshots showed).
    Stripping leading whitespace and dropping every blank line fixes both.
    """
    lines = [line.strip() for line in html.strip().splitlines()]
    flat = "\n".join(line for line in lines if line)
    st.markdown(flat, unsafe_allow_html=True)


def inject_global_css() -> None:
    """Inject the shared fonts + CSS skin. Safe to call on every page."""
    render_html(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
        :root {{
            --forest: {FOREST}; --forest-dark: {FOREST_DARK}; --leaf: {LEAF};
            --cream: {CREAM}; --paper: {PAPER}; --ink: {INK};
            --paprika: {PAPRIKA}; --paprika-dark: {PAPRIKA_DARK};
            --wheat: {WHEAT}; --line: {LINE};
        }}

        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        .stApp {{ background: var(--cream); color: var(--ink); }}

        h1, h2, h3, .bg-display {{
            font-family: 'Fraunces', serif !important;
            color: var(--forest) !important;
            letter-spacing: -0.01em;
        }}

        /* Sidebar: deep awning green with cream text */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, var(--forest) 0%, var(--forest-dark) 100%);
        }}
        [data-testid="stSidebar"] * {{ color: var(--cream) !important; }}
        [data-testid="stSidebar"] h1 {{ font-size: 1.4rem !important; }}
        [data-testid="stSidebar"] hr {{ border-color: rgba(250,246,236,0.18); }}
        [data-testid="stSidebar"] .stButton>button {{
            background: var(--paprika); color: #fff !important; border: none;
            border-radius: 999px; font-weight: 600;
        }}
        [data-testid="stSidebar"] .stButton>button:hover {{ background: var(--paprika-dark); }}
        [data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background: rgba(250,246,236,0.10); border-radius: 8px;
        }}

        /* Buttons everywhere else */
        .stButton>button {{
            border-radius: 999px; border: 1.5px solid var(--forest);
            color: var(--forest); font-weight: 600; background: var(--paper);
        }}
        .stButton>button:hover {{ background: var(--forest); color: #fff !important; border-color: var(--forest); }}

        /* Metrics rendered like little produce-crate tags */
        [data-testid="stMetric"] {{
            background: var(--paper); border: 1px solid var(--line);
            border-left: 5px solid var(--leaf); border-radius: 10px;
            padding: 0.9rem 1rem;
        }}
        [data-testid="stMetricLabel"] {{ color: var(--ink) !important; opacity: 0.7; }}
        [data-testid="stMetricValue"] {{ color: var(--forest) !important; font-family: 'Fraunces', serif; }}

        /* Tabs, expanders, dataframes: soften into the palette */
        .streamlit-expanderHeader {{ font-weight: 600; color: var(--forest); }}
        [data-testid="stDataFrame"] {{ border: 1px solid var(--line); border-radius: 10px; }}

        /* Page header block ("shelf tag") used by theme.page_header() */
        .bg-header {{
            background: var(--paper); border: 1px solid var(--line);
            border-left: 6px solid var(--paprika); border-radius: 12px;
            padding: 1.1rem 1.4rem; margin-bottom: 1.4rem;
            display: flex; align-items: center; gap: 0.9rem;
        }}
        .bg-header .bg-icon {{
            width: 46px; height: 46px; min-width: 46px;
            border-radius: 50%; background: var(--cream);
            display: flex; align-items: center; justify-content: center;
        }}
        .bg-tag svg {{ margin-right: 2px; }}
        .bg-header .bg-title {{ font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.6rem; color: var(--forest); margin: 0; line-height: 1.15; }}
        .bg-header .bg-subtitle {{ margin: 0.15rem 0 0 0; color: var(--ink); opacity: 0.75; font-size: 0.95rem; }}

        /* Hero banner on the landing page */
        .bg-hero {{
            background: linear-gradient(120deg, var(--forest) 0%, var(--forest-dark) 100%);
            background-image: linear-gradient(120deg, rgba(31,77,61,0.96) 0%, rgba(22,58,45,0.97) 100%), url("{_PATTERN_SVG}");
            background-size: cover, 120px 120px;
            border-radius: 18px; padding: 2.6rem 2.4rem; margin-bottom: 1.6rem;
            position: relative; overflow: hidden;
        }}
        .bg-hero::before {{
            content: ""; position: absolute; inset: 0;
            background: repeating-linear-gradient(45deg, var(--wheat) 0 22px, transparent 22px 44px);
            opacity: 0.10;
        }}
        .bg-hero .bg-eyebrow {{
            display: inline-block; color: var(--wheat); font-weight: 600;
            font-size: 0.8rem; letter-spacing: 0.02em; position: relative; z-index: 1;
        }}
        .bg-hero h1 {{
            color: var(--cream) !important; font-size: 2.6rem; margin: 0.3rem 0 0.6rem 0;
            position: relative; z-index: 1;
        }}
        .bg-hero p {{
            color: var(--cream) !important; opacity: 0.9; max-width: 62ch;
            font-size: 1.02rem; line-height: 1.55; position: relative; z-index: 1;
        }}

        /* Feature tags styled like little price/shelf tags */
        .bg-tag-row {{ display: flex; flex-wrap: wrap; gap: 0.6rem; margin-top: 1.1rem; }}
        .bg-tag {{
            background: rgba(250,246,236,0.12); color: var(--cream);
            border: 1px solid rgba(250,246,236,0.35);
            border-radius: 999px; padding: 0.35rem 0.9rem; font-size: 0.85rem;
            position: relative; z-index: 1;
        }}
        </style>
        """
    )


def hero(title: str, tagline: str, eyebrow: str = "AI SUPERMART OPERATIONS ASSISTANT",
         tags: list[str] | None = None, tag_icons: list[str] | None = None) -> None:
    """Render the landing-page hero banner. Presentation only."""
    tags_html = ""
    if tags:
        icons = tag_icons or [None] * len(tags)
        chips = "".join(
            f'<span class="bg-tag">{icon_svg(ic, 15, "var(--wheat)") if ic else ""} {t}</span>'
            for t, ic in zip(tags, icons)
        )
        tags_html = f'<div class="bg-tag-row">{chips}</div>'
    render_html(
        f"""
        <div class="bg-hero">
            <span class="bg-eyebrow">{icon_svg('cart', 15, 'var(--wheat)')} {eyebrow}</span>
            <h1>{title}</h1>
            <p>{tagline}</p>
            {tags_html}
        </div>
        """
    )


def page_header(title: str, subtitle: str = "", icon: str = "cart") -> None:
    """Render a styled page header, replacing a plain st.title()/st.caption() pair.

    ``icon`` is a key into the app's icon set (e.g. "chart", "chat", "search",
    "trend", "book", "receipt", "cart"). Falls back to a leaf glyph if unknown.
    """
    subtitle_html = f'<p class="bg-subtitle">{subtitle}</p>' if subtitle else ""
    render_html(
        f"""
        <div class="bg-header">
            <div class="bg-icon">{icon_svg(icon, 24, 'var(--forest)')}</div>
            <div>
                <p class="bg-title">{title}</p>
                {subtitle_html}
            </div>
        </div>
        """
    )
