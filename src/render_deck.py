"""Render the deck to PNGs via PowerPoint COM, for visual inspection.

The deck is laid out with fixed text boxes, so text overflow is invisible to a
structural check. Rendering every slide and looking at it is the only way to know
the layout holds.

Usage:  python -m src.render_deck [--pdf]
Writes: docs/render/slide_NN.png (and PRESENTATION.pdf with --pdf)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PPTX = ROOT / "docs" / "PRESENTATION.pptx"
OUTDIR = ROOT / "docs" / "render"


def render(to_pdf: bool = False) -> None:
    import win32com.client

    OUTDIR.mkdir(parents=True, exist_ok=True)
    app = win32com.client.Dispatch("PowerPoint.Application")
    try:
        pres = app.Presentations.Open(str(PPTX), WithWindow=False)
        try:
            pres.SaveCopyAs(str(OUTDIR / "presentation"), 18)  # 18 = ppSaveAsPNG
        finally:
            pres.Close()
        if to_pdf:
            pres2 = app.Presentations.Open(str(PPTX), WithWindow=False)
            try:
                pres2.SaveAs(str(ROOT / "docs" / "PRESENTATION.pdf"), 32)  # 32 = ppSaveAsPDF
            finally:
                pres2.Close()
    finally:
        app.Quit()

    # PowerPoint writes slideN.PNG into a folder named after the export
    made = sorted(OUTDIR.rglob("*.PNG")) + sorted(OUTDIR.rglob("*.png")) + sorted(OUTDIR.rglob("*.JPG"))
    print(f"rendered {len(made)} images into {OUTDIR}")
    for m in made[:20]:
        print("  ", m.relative_to(ROOT))


if __name__ == "__main__":
    render("--pdf" in sys.argv)
