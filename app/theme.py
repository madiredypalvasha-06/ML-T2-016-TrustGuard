"""Instrument-panel design system shared by all app pages.

Deliberate aesthetic: a dark "mission control" instrument panel. Big
monospaced readouts, status lights, dials, and carefully ordered color:
  trust  -> emerald
  review -> amber
  abstain / alarm -> red
Data accents stay in a restrained sky-blue / violet pair.
"""
import numpy as np
import plotly.graph_objects as go

# ---- palette --------------------------------------------------------------
BG = "#0B0F17"; PANEL = "#111826"; PANEL2 = "#0E1420"
BORDER = "#223048"; TEXT = "#E6EDF7"; MUTED = "#8B98AC"
GREEN = "#34D399"; AMBER = "#F59E0B"; RED = "#F87171"
SKY = "#38BDF8"; VIOLET = "#A78BFA"

GRADIENTS = {"trust": GREEN, "review": AMBER, "abstain": RED}

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# ---- CSS -------------------------------------------------------------------
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="st-"] {{
    font-family: 'Space Grotesk', sans-serif;
    color: {TEXT};
}}
.block-container {{
    padding-top: 1.1rem; padding-bottom: 3rem;
    max-width: 1500px;
}}
#MainMenu, footer {{ visibility: hidden; }}

.sysbar {{
    display:flex; align-items:baseline; gap:14px;
    border-bottom:1px solid {BORDER}; padding-bottom:10px; margin-bottom:18px;
}}
.sysbar .id {{
    font-family:'IBM Plex Mono', monospace; font-weight:600;
    color:{SKY}; letter-spacing:.12em; font-size:.82rem;
}}
.sysbar .sys-title {{
    font-size:1.35rem; font-weight:700; letter-spacing:.01em;
}}
.sysbar .state {{
    margin-left:auto; font-family:'IBM Plex Mono', monospace;
    font-size:.72rem; color:{MUTED};
}}

.mission {{
    border:1px solid {BORDER}; border-radius:14px;
    background:linear-gradient(160deg, {PANEL} 0%, {PANEL2} 100%);
    padding:22px 26px; margin:10px 0 18px 0;
}}
.mission .lead {{ font-size:1.02rem; line-height:1.5; color:{TEXT}; }}
.mission .lead b {{ color:{SKY}; font-weight:600; }}
.mission .lead a, .mission .lead code {{
    color:{GREEN}; font-family:'IBM Plex Mono', monospace; font-size:.85rem;
}}

.readout {{
    font-family:'IBM Plex Mono', monospace;
    background:{PANEL2}; border:1px solid {BORDER}; border-radius:10px;
    padding:10px 14px;
}}
.readout .k {{ color:{MUTED}; font-size:.66rem; letter-spacing:.14em; text-transform:uppercase; }}
.readout .v {{ font-size:1.14rem; color:{TEXT}; font-weight:600; }}

.panel {{
    border:1px solid {BORDER}; border-radius:12px;
    background:{PANEL}; padding:14px 16px; margin-top:6px;
}}
.panel h3 {{
    font-size:.78rem; letter-spacing:.16em; text-transform:uppercase;
    color:{MUTED}; font-weight:600; margin:0 0 8px 0;
}}

.verdict {{
    border-radius:12px; padding:16px 20px; margin-top:10px;
    font-weight:600; text-align:center; font-size:1.25rem; letter-spacing:.02em;
}}
.verdict.pass  {{ background:rgba(52,211,153,.12); border:1px solid {GREEN}; color:{GREEN}; }}
.verdict.warn  {{ background:rgba(245,158,11,.12); border:1px solid {AMBER}; color:{AMBER}; }}
.verdict.fail  {{ background:rgba(248,113,113,.12); border:1px solid {RED}; color:{RED}; }}

.check-row {{ display:flex; align-items:center; gap:10px; padding:7px 4px; }}
.dot {{ width:12px; height:12px; border-radius:50%; flex:none; }}
.dot.ok  {{ background:{GREEN}; box-shadow:0 0 10px {GREEN}; }}
.dot.mid {{ background:{AMBER}; box-shadow:0 0 10px {AMBER}; }}
.dot.bad {{ background:{RED}; box-shadow:0 0 10px {RED}; }}
.check-row .nm {{ font-family:'IBM Plex Mono', monospace; width:140px; color:{MUTED}; }}
.check-row .ms {{ font-family:'IBM Plex Mono', monospace; color:{TEXT}; }}

.stTabs [data-baseweb="tab"] {{ color:{MUTED}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{SKY}; }}
div[data-testid="stMetric"] {{
    background:{PANEL}; border:1px solid {BORDER}; border-radius:10px; padding:8px 12px;
}}
</style>
"""


def sysbar(page_name, state_html="STANDBY"):
    return (
        '<div class="sysbar">'
        f'<span class="id">ML-T2-016&nbsp;·&nbsp;WHEN-NOT-TO-TRUST</span>'
        f'<span class="sys-title">{page_name}</span>'
        f'<span class="state">{state_html}</span>'
        "</div>"
    )


def mission(html):
    return f'<div class="mission">{html}</div>'


def check_row(light, name, msg):
    cls = {"ok": "ok", "mid": "mid", "bad": "bad"}[light]
    return (f'<div class="check-row"><span class="dot {cls}"></span>'
            f'<span class="nm">{name}</span><span class="ms">{msg}</span></div>')


def plotly_layout(fig, height=300, margin=None, title=None):
    """Apply the shared dark plotly styling."""
    fig.update_layout(
        template=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono, monospace", color=TEXT, size=12),
        height=height,
        margin=margin or dict(l=10, r=10, t=46, b=10),
        title=dict(text=title or "", x=0.5, xanchor="center",
                   font=dict(size=13, color=MUTED)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=11)),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDER, font_color=TEXT),
    )
    fig.update_xaxes(gridcolor="#1B2638", zerolinecolor="#1B2638", tickcolor=BORDER)
    fig.update_yaxes(gridcolor="#1B2638", zerolinecolor="#1B2638", tickcolor=BORDER)
    return fig


def status_light(hex_color, size=16):
    return f'<span style="display:inline-block;width:{size}px;height:{size}px;' \
           f'border-radius:50%;background:{hex_color};' \
           f'box-shadow:0 0 {size//2}px {hex_color};"></span>'


def gauge(fig_title, value, upper, color):
    """Half-circle gauge used for trust dials."""
    value = float(value) if np.isfinite(value) else None
    fig = go.Figure(go.Indicator(
        mode="gauge+number" if value is not None else "gauge",
        value=value if value is not None else 0.0,
        number=dict(suffix="", font=dict(color=TEXT, size=34)),
        gauge=dict(
            axis=dict(range=[0, upper], tickwidth=1, tickcolor=BORDER,
                      tickfont=dict(color=MUTED, size=10)),
            bar=dict(color=color, thickness=0.32),
            bgcolor=PANEL2,
            borderwidth=1, bordercolor=BORDER,
            steps=[dict(range=[0, upper], color=PANEL2)],
            threshold=dict(
                line=dict(color=BORDER, width=2),
                thickness=0.9, value=value if value is not None else 0.0),
        ),
        domain=dict(x=[0, 1], y=[0.25, 1]),
    ))
    if value is None:
        fig.add_annotation(text="n/a", xref="paper", yref="paper",
                           x=0.5, y=0.62, showarrow=False,
                           font=dict(color=MUTED, size=26))
    return plotly_layout(fig, height=210, title=fig_title,
                         margin=dict(l=12, r=12, t=34, b=8))


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)