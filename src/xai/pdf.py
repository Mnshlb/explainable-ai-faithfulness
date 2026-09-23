"""Render the generated report to a print-ready PDF.

The HTML produced by ``xai.report`` is the build intermediate; this module turns
it into the deliverable. WeasyPrint is used rather than a headless browser
because it implements CSS Paged Media — margin boxes, ``string-set`` and
``target-counter`` — which is what gives the document running heads and a table
of contents carrying real page numbers. Chrome's print-to-PDF supports none of
those, and would reduce the report to a web page on paper.

Relative image sources resolve against the document's own location: the report
embeds saliency panels as ``../artifacts/<task>/explanations/<id>.png`` from
``Report/``. Passing the HTML path as the base URL is therefore correct, and
overriding it with the project root would silently drop every panel.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from . import config as cfgmod

HTML_NAME = "ExAI_Report_v2.html"
PDF_NAME = "ExAI_Report.pdf"


def render(html_path: Path, pdf_path: Path) -> Path:
    """Rasterise nothing, vectorise everything: HTML -> PDF.

    Raises a clear error rather than a library traceback when WeasyPrint's
    system libraries (pango/cairo) are missing, since that is the one failure
    a `pip install` alone does not prevent.
    """
    try:
        from weasyprint import HTML
    except OSError as err:  # missing libgobject/pango — not a Python-level problem
        raise SystemExit(
            f"WeasyPrint could not load its system libraries: {err}\n"
            "Install them with:  brew install pango        (macOS)\n"
            "                    apt-get install libpango-1.0-0 libpangoft2-1.0-0  (Debian)"
        ) from err
    except ModuleNotFoundError as err:
        raise SystemExit(
            "WeasyPrint is not installed. Install it with:\n"
            "    pip install weasyprint\n"
            "and its system libraries (brew install pango)."
        ) from err

    if not html_path.exists():
        raise SystemExit(f"no report to convert at {html_path} — run `python -m xai.report` first")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    doc = HTML(filename=str(html_path)).render()
    doc.write_pdf(str(pdf_path))
    n_pages = len(doc.pages)
    size_mb = pdf_path.stat().st_size / 1e6
    print(f"wrote {pdf_path}  ({n_pages} pages, {size_mb:.1f} MB, {time.time() - t0:.1f}s)")
    return pdf_path


def export_figures(cfg, out_dir: Path) -> list[Path]:
    """Write the headline figures as standalone SVGs, for the README.

    GitHub renders static SVG in markdown, so the two figures that carry the
    project's findings can appear above the fold without a rasterisation step.
    """
    import json as _json

    from . import figures

    def _load(name):
        p = cfg.artifacts / name
        return _json.loads(p.read_text()) if p.exists() else {}

    faith, beh = _load("faithfulness.json"), _load("behaviour.json")
    wanted = {
        "faithfulness.svg": figures.faithfulness_bars(faith, "comprehensiveness"),
        "budget-vs-reliability.svg": figures.budget_vs_reliability(
            (beh.get("cross") or {}).get("budget") or {}),
        "token-composition.svg": figures.token_composition(beh),
        "error-detection.svg": figures.error_detection(beh),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, html in wanted.items():
        if not html or "<svg" not in html:
            continue
        svg = html[html.index("<svg"):html.index("</svg>") + 6]
        # A standalone SVG needs its own namespace; inside the report the host
        # document supplies it.
        svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
        dest = out_dir / name
        dest.write_text(svg, encoding="utf-8")
        written.append(dest)
        print(f"  wrote {dest.relative_to(cfgmod.PROJECT_ROOT)}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    ap.add_argument("--open", action="store_true", help="open the PDF when it is written")
    ap.add_argument("--figures", action="store_true",
                    help="also export the headline figures as standalone SVGs")
    args = ap.parse_args()

    cfg = cfgmod.load(args.config)
    report_dir = cfgmod.PROJECT_ROOT / "Report"
    out = render(report_dir / HTML_NAME, report_dir / PDF_NAME)

    if args.figures:
        export_figures(cfg, report_dir / "figures")

    if args.open and sys.platform == "darwin":
        subprocess.run(["open", str(out)], check=False)


if __name__ == "__main__":
    main()
