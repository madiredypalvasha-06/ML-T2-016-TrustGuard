#!/usr/bin/env python3
"""Render Technical_Paper_ML-T2-016.md to a print-quality PDF via headless Chromium.

Usage:  python3 scripts/render_paper_pdf.py
Output: Technical_Paper_ML-T2-016.pdf  (A4, numbered pages, running header, TOC)
"""
import os
import re
import sys
import pathlib

import markdown
from playwright.sync_api import sync_playwright

BASE = pathlib.Path(__file__).resolve().parent.parent
SRC = BASE / "Technical_Paper_ML-T2-016.md"
OUT = BASE / "Technical_Paper_ML-T2-016.pdf"
HTML_TMP = BASE / ".paper_render.html"

CSS = r"""
@page {
  size: A4;
  margin: 20mm 18mm 18mm 18mm;
}
@page :first { margin-top: 16mm; }

html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }

body {
  font-family: "Charter", "Bitstream Charter", "Iowan Old Style",
               "Palatino Linotype", Palatino, Georgia, serif;
  font-size: 9.6pt;
  line-height: 1.52;
  color: #14181f;
  margin: 0;
  text-align: justify;
  hyphens: auto;
  orphans: 3;
  widows: 3;
}

/* ---------- running header / footer ----------
   Supplied by Chromium's native header/footer templates, not by the body. */

/* ---------- title block ---------- */
h1 {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 20pt; line-height: 1.2; font-weight: 700;
  letter-spacing: -.015em; color: #0b1220;
  margin: 0 0 6mm 0; padding: 0 0 4mm 0;
  border-bottom: 2.2pt solid #111827;
  text-align: left; hyphens: none;
}
h1 + p { margin-top: 0; }

/* ---------- headings ---------- */
h2, h3, h4 {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  color: #0b1220; text-align: left; hyphens: none;
  page-break-after: avoid; break-after: avoid-page;
}
h2 {
  font-size: 12.4pt; font-weight: 700; letter-spacing: -.01em;
  margin: 9mm 0 3.5mm 0; padding-bottom: 1.6mm;
  border-bottom: .7pt solid #c9d1dc;
}
h3 { font-size: 10.4pt; font-weight: 650; margin: 6mm 0 2.4mm 0; color: #1f2a3a; }
h4 { font-size: 9.6pt; font-weight: 650; margin: 4.5mm 0 2mm 0; color: #374151; }

/* ---------- body ---------- */
p { margin: 0 0 2.7mm 0; orphans: 3; widows: 3; }
strong { color: #05070c; font-weight: 680; }
em { font-style: italic; }
a { color: #1d4ed8; text-decoration: none; }

ul, ol { margin: 0 0 3mm 0; padding-left: 6.5mm; }
li { margin-bottom: 1.1mm; }
li > ul, li > ol { margin-top: 1.1mm; }

/* ---------- tables ---------- */
table {
  width: 100%;
  border-collapse: collapse;
  margin: 3mm 0 4mm 0;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 8.1pt;
  page-break-inside: avoid;
  break-inside: avoid;
  text-align: left;
}
thead { display: table-header-group; }
tr { page-break-inside: avoid; break-inside: avoid; }
th {
  background: #eef2f7;
  color: #0b1220; font-weight: 680;
  text-align: left; padding: 1.5mm 2mm;
  border-top: 1.1pt solid #94a3b8;
  border-bottom: .6pt solid #94a3b8;
}
td {
  padding: 1.35mm 2mm;
  border-bottom: .35pt solid #e2e8f0;
  vertical-align: top;
}
tbody tr:nth-child(even) td { background: #fafbfc; }
tbody tr:last-child td { border-bottom: 1.1pt solid #94a3b8; }

/* numeric columns read better centred/right; keep first col left */
td:not(:first-child), th:not(:first-child) { text-align: center; }

/* ---------- code ---------- */
code {
  font-family: "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
  font-size: .86em; background: #f1f4f8;
  padding: .4mm 1.1mm; border-radius: 1.6pt;
  color: #1e293b; word-break: break-word;
}
pre {
  background: #f7f9fb; border: .5pt solid #dde3ea;
  border-left: 2.2pt solid #94a3b8;
  border-radius: 2.5pt;
  padding: 2.6mm 3mm; margin: 3mm 0;
  font-size: 7.7pt; line-height: 1.42;
  page-break-inside: avoid; break-inside: avoid;
  text-align: left; white-space: pre-wrap; word-wrap: break-word;
}
pre code { background: none; padding: 0; font-size: 1em; }

/* ---------- figures ---------- */
figure {
  margin: 4.5mm 0 5mm 0; text-align: center;
  page-break-inside: avoid; break-inside: avoid;
}
figure img {
  width: 100%; max-width: 158mm; height: auto;
  border: .5pt solid #dde3ea; border-radius: 2pt;
  background: #fff;
}
figcaption {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 7.9pt; line-height: 1.42; color: #475569;
  text-align: left; margin-top: 2mm; hyphens: none;
}
figcaption em { color: #334155; }

/* ---------- blockquote / abstract ---------- */
blockquote {
  margin: 3.5mm 0; padding: 2.6mm 3.4mm;
  background: #f6f8fa; border-left: 2.2pt solid #64748b;
  font-size: 9.1pt; page-break-inside: avoid;
}
blockquote p:last-child { margin-bottom: 0; }

/* ---------- abstract block ---------- */
.abstract {
  background: #f6f8fa;
  border: .5pt solid #dde3ea; border-left: 2.6pt solid #0f172a;
  border-radius: 2pt;
  padding: 3.4mm 4mm; margin: 0 0 5mm 0;
  font-size: 9.2pt;
}
.abstract > p:last-child { margin-bottom: 0; }

/* ---------- keywords / metadata ---------- */
table.meta td { border: none; padding: .8mm 0; font-size: 8.6pt; }
table.meta td:first-child {
  width: 34mm; color: #64748b; text-transform: uppercase;
  letter-spacing: .05em; font-size: 7.2pt; font-weight: 650;
  text-align: left; padding-top: 1.2mm;
}
table.meta { margin: 0 0 5mm 0; border-bottom: .8pt solid #c9d1dc; }
table.meta tr:last-child td { border-bottom: none; }

/* ---------- table of contents ---------- */
.toc { font-size: 8.9pt; margin: 0 0 6mm 0; }
.toc ul { list-style: none; padding-left: 0; margin: 0; }
.toc li { margin: 0; padding: .7mm 0; border-bottom: .3pt dotted #dbe1e8; }
.toc li.lvl2 { padding-left: 6mm; font-size: 8.2pt; color: #475569; }
.toc li.lvl2:last-child, .toc li.lvl1:last-child { border-bottom: none; }

/* ---------- misc ---------- */
hr { border: none; border-top: .5pt solid #d7dce3; margin: 6mm 0; }
.mdn { text-align: right; color: #94a3b8; font-size: 7.6pt; margin-top: 8mm;
       font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
strong.mdk { background: #fff3bf; padding: 0 .6mm; border-radius: 1.2pt; }
"""

JS = r"""
// Tell the driver when MathJax has finished typesetting so it can print a
// fully-rendered document rather than a half-typeset one.
(function () {
  function ready() { window.__mathjaxDone = true; }
  if (window.MathJax && window.MathJax.startup && window.MathJax.startup.promise) {
    window.MathJax.startup.promise.then(function () {
      // one extra frame so the browser has laid out the new SVG nodes
      requestAnimationFrame(function () { requestAnimationFrame(ready); });
    });
  } else {
    window.addEventListener('load', ready);
  }
})();
"""


MATH_TOKEN = "MATH{}"


def extract_math(md_text: str) -> tuple[str, list[str]]:
    """Pull LaTeX out of the markdown before the converter can mangle it.

    Markdown treats `_` and `*` as emphasis, so an expression such as
    $\\mathrm{acc}_b$ loses its subscript before MathJax ever sees it. We
    therefore swap every math span for an opaque placeholder, run the
    markdown conversion, and put the LaTeX back afterwards.
    """
    spans: list[str] = []

    def stash(m: re.Match) -> str:
        spans.append(m.group(0))
        return f"{MATH_TOKEN}{len(spans) - 1}{MATH_TOKEN}"

    # Fenced code blocks are literal; never touch their contents.
    blocks: list[str] = []

    def stash_block(m: re.Match) -> str:
        blocks.append(m.group(0))
        return f"CODEBLOCK{len(blocks) - 1}"

    text = re.sub(r"```.*?```", stash_block, md_text, flags=re.S)
    text = re.sub(r"\$\$(.+?)\$\$", stash, text, flags=re.S)
    text = re.sub(r"(?<!\$)\$([^\$\n]+?)\$(?!\$)", stash, text)
    text = text.replace("$$", "$$")
    return text, spans, blocks


def build_html(md_text: str) -> str:
    """Markdown -> HTML, with the paper's specific conventions handled."""
    protected, math_spans, code_blocks = extract_math(md_text)

    html_body = markdown.markdown(
        protected,
        extensions=["tables", "fenced_code", "attr_list", "sane_lists", "md_in_html"],
        output_format="html5",
    )

    # Restore code blocks first: they may themselves contain math-looking text.
    for i, block in enumerate(code_blocks):
        inner = block[len("```") : -len("```")]
        first, _, rest = inner.partition("\n")
        lang = first.strip()
        body = rest if rest else first
        esc = (
            body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        cls = f' class="language-{lang}"' if lang.isalnum() else ""
        html_body = html_body.replace(
            f"CODEBLOCK{i}", f"<pre><code{cls}>{esc.strip()}\n</code></pre>"
        )

    # Restore LaTeX as MathJax delimiters.
    for i, expr in enumerate(math_spans):
        if expr.startswith("$$"):
            html_body = html_body.replace(
                f"{MATH_TOKEN}{i}{MATH_TOKEN}", "\\[" + expr[2:-2].strip() + "\\]"
            )
        else:
            html_body = html_body.replace(
                f"{MATH_TOKEN}{i}{MATH_TOKEN}", "\\(" + expr[1:-1].strip() + "\\)"
            )

    # --- abstract ------------------------------------------------------
    # The paper has "## Abstract"; wrap its content in a tinted block.
    m = re.search(
        r'(<h2[^>]*>Abstract</h2>)(.*?)(?=<h2[^>]*>)', html_body, re.S
    )
    if m:
        html_body = (
            html_body[: m.start()]
            + m.group(1)
            + '<div class="abstract">' + m.group(2).strip() + "</div>"
            + html_body[m.end():]
        )

    # --- table of contents ---------------------------------------------
    # Replace the markdown-generated TOC list with a styled one; page numbers
    # are omitted because CSS paged-media counters are not reliably exposed.
    m = re.search(r'(<h2[^>]*>Table of Contents</h2>)\s*(<ul>.*?</ul>)', html_body, re.S)
    if m:
        items = re.findall(r"<li>(.*?)</li>", m.group(2), re.S)
        lis = []
        for it in items:
            clean = re.sub(r"<[^>]+>", "", it).replace("&amp;", "&").strip()
            lvl = "lvl2" if clean[:1].isdigit() and "." in clean[:4] else "lvl1"
            lis.append(f'<li class="{lvl}">{clean}</li>')
        html_body = (
            html_body[: m.start()]
            + m.group(1)
            + '<div class="toc"><ul>' + "".join(lis) + "</ul></div>"
            + html_body[m.end():]
        )

    # --- metadata table -> class="meta" --------------------------------
    html_body = html_body.replace("<table>", '<table class="meta">', 1)

    # --- figures: wrap each img in <figure> + build caption ------------
    def fig(m):
        src, alt = m.group(1), m.group(2)
        # the following <p><em>...</em></p> is the hand-written caption
        return (
            f'<figure><img src="{src}" alt="{alt}">'
            f"<!--CAPTION--></figure>"
        )

    html_body = re.sub(
        r'<p><img src="([^"]+)" alt="([^"]*)"\s*/?></p>', fig, html_body
    )

    # The captions are separate italic paragraphs directly after each figure.
    # Attach the immediately-following italic paragraph to the figure.
    parts = html_body.split("<!--CAPTION-->")
    if len(parts) > 1:
        rebuilt = [parts[0]]
        for chunk in parts[1:]:
            mcap = re.match(
                r"\s*<p><em>(.*?)</em></p>", chunk, re.S
            )
            if mcap:
                rebuilt.append(
                    f'<figcaption>{mcap.group(1)}</figcaption></figure>'
                    + chunk[mcap.end():]
                )
            else:
                rebuilt.append("</figure>" + chunk)
        html_body = "".join(rebuilt)

    # --- end-of-document note ------------------------------------------
    html_body = html_body.replace(
        "<p><em>End of report.", '<p class="mdn"><em>End of report.'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ML-T2-016 — Teaching a Machine Learning System When Not to Trust Its Own Prediction</title>
<style>{CSS}</style>
<script>
window.MathJax = {{
  tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] }},
  svg: {{ fontCache: 'none' }},
  options: {{ skipHtmlTags: ['script','noscript','style','textarea','pre','code'] }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-svg.js"></script>
<script>{JS}</script>
<style>
mjx-container {{ font-size: 97% !important; }}
mjx-container[display="true"] {{ margin: 1.1em 0 !important; }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""


def count_pdf_pages(path: pathlib.Path) -> int:
    """Number of pages in a Chromium-produced PDF, read from the page tree."""
    blob = path.read_bytes()
    n = len(re.findall(rb"/Type\s*/Page[^s]", blob))
    return max(1, n)


def main() -> int:
    md_text = SRC.read_text(encoding="utf-8")
    html = build_html(md_text)
    HTML_TMP.write_text(html, encoding="utf-8")
    print(f"[render] wrote {HTML_TMP.name} ({len(html)/1024:.0f} KB)")

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--font-render-hinting=none"])
        page = browser.new_page()
        page.goto(HTML_TMP.resolve().as_uri(), wait_until="load", timeout=120_000)

        # Block until MathJax has actually typeset the document.
        try:
            page.wait_for_function("window.__mathjaxDone === true", timeout=90_000)
        except Exception:
            print("[render] warning: MathJax did not signal ready; continuing")
        page.wait_for_timeout(2000)

        n_math = page.evaluate("document.querySelectorAll('mjx-container').length")
        n_img = page.evaluate(
            "Array.from(document.images).filter(i => i.naturalWidth > 0).length"
        )
        print(f"[render] math expressions typeset: {n_math}")
        print(f"[render] figures loaded         : {n_img}")
        if n_img < 5:
            print(f"[render] WARNING: only {n_img}/5 figures resolved")

        def emit() -> None:
            page.pdf(
                path=str(OUT),
                format="A4",
                print_background=True,
                display_header_footer=True,
                # Chromium substitutes <span class="pageNumber"> and
                # <span class="totalPages"> itself, so a single pass is exact.
                header_template=(
                    '<div style="width:100%;font-family:Helvetica,Arial,sans-serif;'
                    "font-size:7.2pt;color:#6b7280;padding:0 18mm;"
                    "display:flex;justify-content:space-between;"
                    'border-bottom:0.4pt solid #d7dce3;">'
                    "<span>ML-T2-016 &middot; Teaching a Machine Learning System "
                    "When Not to Trust Its Own Prediction</span>"
                    "<span>Technical Report &middot; September 2026</span>"
                    "</div>"
                ),
                footer_template=(
                    '<div style="width:100%;font-family:Helvetica,Arial,sans-serif;'
                    "font-size:7.2pt;color:#6b7280;padding:0 18mm;"
                    "display:flex;justify-content:space-between;"
                    'border-top:0.4pt solid #d7dce3;">'
                    "<span>LearnDepth Academy LLP &middot; Track 2 Advanced ML "
                    "Internship</span>"
                    '<span>Page <span class="pageNumber"></span> of '
                    '<span class="totalPages"></span></span>'
                    "</div>"
                ),
                margin={
                    "top": "20mm",
                    "bottom": "16mm",
                    "left": "18mm",
                    "right": "18mm",
                },
            )

        emit()
        total = count_pdf_pages(OUT)
        browser.close()

    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"[render] wrote {OUT.name}  ({total} pages, {size_mb:.2f} MB)")
    if size_mb > 10:
        print("[render] WARNING: exceeds the 10 MB upload limit")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
