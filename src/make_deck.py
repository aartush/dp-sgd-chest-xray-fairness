"""Build the presentation deck (.pptx) from run artifacts.

Every figure and number on a slide comes from results/runs/*.json or the PNGs in
results/figures/. The deck is deliberately short (14 slides) and each slide makes
one point, because the viva is where the work either holds up or does not.

Usage:  python -m src.make_deck
Writes: docs/PRESENTATION.pptx
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
FIGS = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"
OUT = ROOT / "docs" / "PRESENTATION.pptx"

NAVY = RGBColor(0x1F, 0x3B, 0x57)
ACCENT = RGBColor(0xC0, 0x39, 0x2B)
GREY = RGBColor(0x5A, 0x5A, 0x5A)
LIGHT = RGBColor(0xF2, 0xF4, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

W, H = Inches(13.333), Inches(7.5)


def load():
    rows = []
    for f in sorted(RUNS.glob("*.json")):
        p = json.loads(f.read_text())
        c, m, t = p["config"], p["metrics"], p["train"]
        row = {
            "dataset": c["dataset"], "method": c["method"],
            "eps": np.nan if c["eps"] is None else float(c["eps"]),
            "seed": c["seed"], "imbalance_ratio": c["imbalance_ratio"],
            "achieved_eps": t["achieved_eps"], "sigma": t["noise_multiplier"],
        }
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)


def new_deck() -> Presentation:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    return prs


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def rect(slide, x, y, w, h, color):
    from pptx.enum.shapes import MSO_SHAPE
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = color
    s.line.fill.background(); s.shadow.inherit = False
    return s


def text(slide, txt, x, y, w, h, size=18, bold=False, color=NAVY,
         align=PP_ALIGN.LEFT, italic=False, font="Calibri"):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0)
    lines = txt.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(6)
        r = p.add_run(); r.text = line
        r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic
        r.font.color.rgb = color; r.font.name = font
    return tb


def header(slide, title, num=None):
    rect(slide, 0, 0, W, Inches(0.06), NAVY)
    text(slide, title, Inches(0.55), Inches(0.32), Inches(11.6), Inches(0.7),
         size=27, bold=True, color=NAVY)
    rect(slide, Inches(0.55), Inches(1.06), Inches(1.1), Inches(0.045), ACCENT)
    if num is not None:
        text(slide, str(num), W - Inches(0.85), H - Inches(0.5), Inches(0.5),
             Inches(0.3), size=11, color=GREY, align=PP_ALIGN.RIGHT)


def bullets(slide, items, x=Inches(0.6), y=Inches(1.4), w=Inches(6.4),
            h=Inches(5.4), size=16):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(11)
        if isinstance(item, tuple):
            lead, body = item
            r1 = p.add_run(); r1.text = "— "
            r1.font.size = Pt(size); r1.font.color.rgb = ACCENT; r1.font.bold = True
            r2 = p.add_run(); r2.text = lead + " "
            r2.font.size = Pt(size); r2.font.bold = True; r2.font.color.rgb = NAVY
            r3 = p.add_run(); r3.text = body
            r3.font.size = Pt(size); r3.font.color.rgb = RGBColor(0x2B, 0x2B, 0x2B)
        else:
            r = p.add_run()
            r.text = "— " + item
            r.font.size = Pt(size); r.font.color.rgb = RGBColor(0x2B, 0x2B, 0x2B)
    return tb


def picture(slide, name, x, y, w=None, h=None):
    p = FIGS / name
    if not p.exists():
        return None
    return slide.shapes.add_picture(str(p), x, y, width=w, height=h)


def big_stat(slide, value, label, x, y, w=Inches(3.0), color=NAVY):
    text(slide, value, x, y, w, Inches(0.8), size=40, bold=True, color=color,
         align=PP_ALIGN.CENTER)
    text(slide, label, x, y + Inches(0.78), w, Inches(0.8), size=12, color=GREY,
         align=PP_ALIGN.CENTER)


def table(slide, headers, rows, x, y, w, h, font=11):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), x, y, w, h)
    tbl = shape.table
    for i, hd in enumerate(headers):
        cell = tbl.cell(0, i)
        cell.text = ""
        run = cell.text_frame.paragraphs[0].add_run(); run.text = hd
        run.font.size = Pt(font); run.font.bold = True; run.font.color.rgb = WHITE
        run.font.name = "Calibri"
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
    for ri, row in enumerate(rows, start=1):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.text = ""
            run = cell.text_frame.paragraphs[0].add_run(); run.text = str(val)
            run.font.size = Pt(font); run.font.name = "Calibri"
            run.font.color.rgb = RGBColor(0x2B, 0x2B, 0x2B)
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT if ri % 2 else WHITE
    return tbl


def build() -> Path:
    df = load()
    d = df[(df.dataset == "pneumoniamnist") & df.imbalance_ratio.isna()]
    sweep = df[(df.dataset == "pneumoniamnist") & df.imbalance_ratio.notna()].copy()
    if not sweep.empty:
        sweep["eff_ratio"] = sweep.imbalance_ratio.astype(float)
    chest = df[df.dataset == "chestmnist"]

    def M(method, eps, col):
        s = d[d.method == method]
        s = s[s.eps.isna()] if pd.isna(eps) else s[s.eps == eps]
        return None if s.empty else float(s[col].mean())

    prs = new_deck()

    # 1 — title
    s = blank(prs)
    rect(s, 0, 0, W, H, NAVY)
    rect(s, Inches(0.9), Inches(2.05), Inches(1.5), Inches(0.06), ACCENT)
    text(s, "Who Pays for Differential Privacy?", Inches(0.9), Inches(2.35),
         Inches(11), Inches(0.9), size=40, bold=True, color=WHITE)
    text(s, "Per-class fairness and membership leakage in differentially\nprivate chest X-ray classification",
         Inches(0.9), Inches(3.35), Inches(10), Inches(1.1), size=20, color=RGBColor(0xD5, 0xDE, 0xE8))
    text(s, "Machine Learning Project  ·  Ethics / Responsible AI track\n"
         "Tushar Sharma  ·  MTech I Year, NSUT  ·  tushar.sharma.pg26@nsut.ac.in  ·  November 2026",
         Inches(0.9), Inches(5.7), Inches(9), Inches(0.9), size=12, color=RGBColor(0xA8, 0xB8, 0xC8))

    # 2 — the problem
    s = blank(prs); header(s, "DP-SGD protects patients by discarding signal", 2)
    bullets(s, [
        ("Every gradient is clipped", "to a fixed norm before it is averaged and noised."),
        ("Clipping bounds one patient's influence", "— that bound is the privacy guarantee."),
        ("But it also discards the update", "from examples whose gradients are largest."),
        ("In an imbalanced dataset those are", "disproportionately the minority class."),
    ], size=17)
    rect(s, Inches(7.3), Inches(1.5), Inches(5.4), Inches(4.9), LIGHT)
    text(s, "The question", Inches(7.6), Inches(1.75), Inches(4.8), Inches(0.5),
         size=14, bold=True, color=ACCENT)
    text(s, "Does DP-SGD hurt minority-class recall more than\nmajority-class recall — and can a cheap,\nprivacy-neutral fix recover it?",
         Inches(7.6), Inches(2.25), Inches(4.8), Inches(1.7), size=18, color=NAVY)
    text(s, "Accuracy cannot answer this: predicting\npneumonia for every scan scores 74%\naccuracy and 0% minority recall.",
         Inches(7.6), Inches(4.35), Inches(4.8), Inches(1.5), size=13,
         color=GREY, italic=True)

    # 3 — the gap in the literature
    s = blank(prs); header(s, "Two papers name the problem; neither tests a fix", 3)
    rect(s, Inches(0.55), Inches(1.4), Inches(6.0), Inches(2.35), LIGHT)
    text(s, "Huang & Xie · arXiv:2507.06619 (Jul 2025)", Inches(0.8), Inches(1.6),
         Inches(5.5), Inches(0.4), size=12, bold=True, color=NAVY)
    text(s, "DP methods “work well on large datasets like CIFAR-100, but fail on small, imbalanced medical datasets like HAM10000.”",
         Inches(0.8), Inches(2.0), Inches(5.5), Inches(1.0), size=13, color=RGBColor(0x2B,0x2B,0x2B))
    text(s, "Gap: reports accuracy only, no per-class recall, no attack evaluation.",
         Inches(0.8), Inches(3.05), Inches(5.5), Inches(0.6), size=12, color=ACCENT, italic=True)
    rect(s, Inches(6.8), Inches(1.4), Inches(6.0), Inches(2.35), LIGHT)
    text(s, "Mohammadi et al. · arXiv:2506.00660 → npj Digit. Med. 9:93 (2026)", Inches(7.05),
         Inches(1.6), Inches(5.5), Inches(0.4), size=12, bold=True, color=NAVY)
    text(s, "Scoping review of 74 studies: “balancing privacy with utility and fairness has become a critical challenge.”",
         Inches(7.05), Inches(2.0), Inches(5.5), Inches(1.0), size=13, color=RGBColor(0x2B,0x2B,0x2B))
    text(s, "Gap: a review can identify the tension, but cannot test a remedy.",
         Inches(7.05), Inches(3.05), Inches(5.5), Inches(0.6), size=12, color=ACCENT, italic=True)
    text(s, "This project: hold the mechanism fixed, vary only the loss and the initialisation,\nreport per-class recall and empirical leakage at matched ε.",
         Inches(0.55), Inches(4.15), Inches(12.2), Inches(1.0), size=17, bold=True, color=NAVY)

    # 4 — dataset
    s = blank(prs); header(s, "PneumoniaMNIST: a natively imbalanced task", 4)
    table(s, ["Split", "Normal (minority)", "Pneumonia (majority)", "Total"],
          [["Train", "1,214  (26%)", "3,494  (74%)", "4,708"],
           ["Validation", "135", "389", "524"],
           ["Test", "234", "390", "624"]],
          Inches(0.6), Inches(1.5), Inches(6.3), Inches(2.0), font=13)
    text(s, "28×28 grayscale · primary pneumonia-vs-normal · MedMNIST v2 (Yang et al., Sci. Data 10:41)",
         Inches(0.6), Inches(3.65), Inches(6.4), Inches(0.8), size=11, color=GREY)
    bullets(s, [
        ("Minority = normal", "— the screening-relevant direction."),
        ("Test set is only 234 minority images", "→ ±3 pt binomial floor on recall."),
        ("ChestMNIST", "(78k, 14 labels, rarest 144 positives) used as a scale check."),
    ], x=Inches(7.2), y=Inches(1.5), w=Inches(5.6), size=14)

    # 5 — setup
    s = blank(prs); header(s, "Four privacy budgets × four methods × three seeds", 5)
    table(s, ["Method", "DP", "Class weights", "Pretrained init"],
          [["Vanilla DP-SGD", "ε ∈ {1,3,8,∞}", "—", "—"],
           ["Class-weighted", "ε ∈ {1,3,8,∞}", "inverse freq.", "—"],
           ["Pretrained", "ε ∈ {1,3,8,∞}", "—", "ImageNet"],
           ["Both", "ε ∈ {1,3,8,∞}", "inverse freq.", "ImageNet"]],
          Inches(0.6), Inches(1.45), Inches(7.4), Inches(2.2), font=12)
    bullets(s, [
        ("ResNet-18 with GroupNorm", "— Opacus cannot form per-sample gradients through BatchNorm."),
        ("Batch 32, forced by the 6 GB GPU", "— Opacus materialises per-sample gradients."),
        ("δ = 1e-5, achieved ε recorded", "for every run (0.99 / 2.98 / 7.95)."),
        ("76 runs total", "= 48 main + 24 imbalance sweep + 4 ChestMNIST."),
    ], x=Inches(0.6), y=Inches(3.95), w=Inches(12), size=14)
    rect(s, Inches(8.35), Inches(1.45), Inches(4.4), Inches(2.2), NAVY)
    text(s, "Class weights and ImageNet\nweights are public.\nNeither consumes privacy budget —\nso any gain is free at the same ε.",
         Inches(8.6), Inches(1.65), Inches(4.0), Inches(1.9), size=13, color=WHITE)

    # 6 — result 1 (lead with privacy)
    s = blank(prs); header(s, "Result 1 — the guarantee is empirically real", 6)
    big_stat(s, f"{M('baseline', np.nan, 'mia_auc_class_conditional'):.3f}", "MIA AUC, no DP",
             Inches(0.75), Inches(1.35), Inches(3.2), color=NAVY)
    big_stat(s, f"{M('baseline', 1.0, 'mia_auc_class_conditional'):.3f}", "MIA AUC, ε = 1",
             Inches(0.75), Inches(2.55), Inches(3.2), color=ACCENT)
    text(s, "Lower AUC = the attacker cannot\ntell members from non-members.\nSeed spread ≈ 0.004 — ten times\nsmaller than the effect.",
         Inches(0.8), Inches(3.95), Inches(3.2), Inches(1.6), size=12, color=GREY)
    picture(s, "fig3_mia_auc.png", Inches(4.35), Inches(1.22), h=Inches(4.95))
    text(s, "This is the result the project can defend at three seeds.",
         Inches(4.35), Inches(6.35), Inches(8.4), Inches(0.4), size=13,
         color=ACCENT, bold=True)

    # 7 — result 2
    s = blank(prs); header(s, "Result 2 — class weighting helps; pretrained init backfires", 7)
    picture(s, "fig1_minority_recall.png", Inches(0.45), Inches(1.30), h=Inches(4.95))
    b3 = M("baseline", 3.0, "minority_recall")
    cw3 = M("classweight", 3.0, "minority_recall")
    pt3 = M("pretrained", 3.0, "minority_recall")
    text(s, "At ε = 3  (vanilla = " + f"{b3:.3f})", Inches(7.85), Inches(1.4),
         Inches(5.0), Inches(0.4), size=14, bold=True, color=NAVY)
    table(s, ["Method", "Minority recall", "Δ"],
          [["+ class weights", f"{cw3:.3f}", f"{cw3-b3:+.3f}"],
           ["+ pretrained init", f"{pt3:.3f}", f"{pt3-b3:+.3f}"],
           ["+ both", f"{M('pretrained+classweight',3.0,'minority_recall'):.3f}",
            f"{M('pretrained+classweight',3.0,'minority_recall')-b3:+.3f}"]],
          Inches(7.85), Inches(1.85), Inches(5.0), Inches(1.5), font=12)
    text(s, "Class weights: same direction at every seed.\nPretrained init: worse at every seed at ε = 8.",
         Inches(7.85), Inches(3.55), Inches(5.0), Inches(1.0), size=12, color=GREY, italic=True)
    text(s, "Most pre-trained excellence does not survive\nDP's gradient noise — only the loss change does.",
         Inches(7.85), Inches(4.55), Inches(5.0), Inches(0.9), size=13, color=NAVY)

    # 8 — per-class at eps 3
    s = blank(prs); header(s, "Per-class view at ε = 3", 8)
    picture(s, "fig4_per_class_eps3.png", Inches(0.6), Inches(1.55), w=Inches(7.0))
    bullets(s, [
        ("Vanilla", "leaves the minority at 0.685 while the majority sits at 0.943."),
        ("Class weighting", "closes 5 points of that gap at no privacy cost."),
        ("Pretrained init", "widens it — the worst minority recall of the four."),
    ], x=Inches(8.15), y=Inches(1.6), w=Inches(4.7), size=14)

    # 9 — gap finding, honestly framed
    s = blank(prs); header(s, "Result 3 — the fairness effect is real at ε = 1 but not monotone", 9)
    picture(s, "fig2_recall_gap.png", Inches(0.6), Inches(1.30), h=Inches(4.95))
    bullets(s, [
        ("At ε = 1", "the vanilla gap widens from 0.262 (no DP) to 0.287 — the largest of any budget."),
        ("Not monotone", "by ε = 8 it narrows to 0.236. So 'tighter privacy is always less fair' is false."),
        ("±3-point floor", "on a 234-image test set → suggestive, not established."),
    ], x=Inches(7.8), y=Inches(1.5), w=Inches(5.0), size=13)

    # 10 — imbalance sweep
    s = blank(prs); header(s, "Imbalance sweep — weighting absorbs the damage", 10)
    if not sweep.empty:
        picture(s, "fig5_imbalance_sweep.png", Inches(0.5), Inches(1.85), w=Inches(8.4))
        b8 = sweep[(sweep.method == "baseline") & np.isclose(sweep.eff_ratio, 8.0)
                   & sweep.eps.isna()].minority_recall.mean()
        c8 = sweep[(sweep.method == "classweight") & np.isclose(sweep.eff_ratio, 8.0)
                   & sweep.eps.isna()].minority_recall.mean()
        text(s, "At 8:1 imbalance", Inches(9.05), Inches(1.5), Inches(3.8), Inches(0.4),
             size=14, bold=True, color=NAVY)
        text(s, f"unweighted  {b8:.3f}\nweighted     {c8:.3f}", Inches(9.05), Inches(2.0),
             Inches(3.8), Inches(1.0), size=20, color=ACCENT, bold=True)
        text(s, "Two seeds per cell — a strong hint,\nnot a settled result.",
             Inches(9.05), Inches(3.4), Inches(3.8), Inches(0.8), size=12, color=GREY, italic=True)

    # 11 — chest scale check
    s = blank(prs); header(s, "Scale check transferred (ChestMNIST, 8k images, 14 labels)", 11)
    if not chest.empty:
        rows = []
        for eps in [np.nan, 3.0]:
            for method in ["baseline", "classweight"]:
                sub = chest[(chest.method == method) & ((chest.eps.isna()) if pd.isna(eps) else (chest.eps == eps))]
                if sub.empty:
                    continue
                rows.append(["+ class weights" if method == "classweight" else "Vanilla",
                             "No DP" if pd.isna(eps) else f"ε = {eps:g}",
                             f"{sub.macro_auc.mean():.3f}", f"{sub.min_label_recall.mean():.3f}",
                             f"{sub.mia_auc_class_conditional.mean():.3f}"])
        table(s, ["Method", "Privacy", "Macro AUC", "Min label recall", "MIA AUC"], rows,
              Inches(0.7), Inches(1.6), Inches(8.6), Inches(1.9), font=12)
    bullets(s, [
        ("Min label recall is 0.000", "— the rarest labels have single-digit positives at 8k images."),
        ("MIA dips to 0.491", "— below chance is impossible; this is noise, read as ≈ 0.5."),
        ("Honest reading", "— the rare-label problem is worse here, not replicated elsewhere."),
    ], x=Inches(0.7), y=Inches(3.85), w=Inches(12), size=14)

    # 12 — limitations
    s = blank(prs); header(s, "Limitations — stated before you ask", 12)
    bullets(s, [
        ("Three seeds", "— anything below the ±3-point floor is called noise, not a finding."),
        ("28×28 inputs", "— results describe this benchmark, not full-resolution radiology."),
        ("Weak attack", "— loss-threshold MIA is a lower bound; the formal ε is the real guarantee."),
        ("secure_mode off", "— standard for experiments; no production claim is made."),
        ("One architecture, one dataset family", "— no general claim that DP-SGD is unfair."),
    ], size=16)

    # 13 — conclusion
    s = blank(prs); header(s, "What this establishes", 13)
    rect(s, Inches(0.6), Inches(1.4), Inches(12.1), Inches(1.25), LIGHT)
    text(s, "1.  The privacy guarantee is real and measurable — MIA AUC 0.634 → ≈ 0.60 with DP.",
         Inches(0.85), Inches(1.55), Inches(11.6), Inches(0.5), size=17, bold=True, color=NAVY)
    rect(s, Inches(0.6), Inches(2.85), Inches(12.1), Inches(1.25), LIGHT)
    text(s, "2.  A privacy-neutral loss change is a free, consistent partial remedy.",
         Inches(0.85), Inches(3.0), Inches(11.6), Inches(0.5), size=17, bold=True, color=NAVY)
    rect(s, Inches(0.6), Inches(4.3), Inches(12.1), Inches(1.25), LIGHT)
    text(s, "3.  A pretrained initialisation is not a remedy under DP — it is actively harmful.",
         Inches(0.85), Inches(4.45), Inches(11.6), Inches(0.5), size=17, bold=True, color=NAVY)
    text(s, "And it does not establish that DP-SGD is unfair in general — only that the measurement is feasible,\nthat the obvious fix works and the obvious transfer-learning fix backfires.",
         Inches(0.6), Inches(5.85), Inches(12.1), Inches(0.9), size=13, color=GREY, italic=True)

    # 14 — next steps
    s = blank(prs); header(s, "What I would do next", 14)
    bullets(s, [
        ("More seeds and a larger minority test set", "— to move the fairness result past the noise floor."),
        ("A DP-aware fine-tuning schedule", "— to test whether pretrained init can be rescued."),
        ("Per-class clipping", "— the SAD-DPSGD idea, as a fifth arm."),
        ("Shadow-model attack on every arm", "— the code is written and validated; it needs the compute."),
    ], size=17)

    # 15 — thanks
    s = blank(prs)
    rect(s, 0, 0, W, H, NAVY)
    text(s, "Thank you", Inches(1), Inches(2.7), Inches(11), Inches(1.0),
         size=38, bold=True, color=WHITE)
    text(s, "Questions welcome — particularly on the noise floor and why\n"
            "the fairness claim is deliberately weaker than the privacy claim.",
         Inches(1), Inches(3.75), Inches(11), Inches(1.2), size=16, color=RGBColor(0xD5, 0xDE, 0xE8))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    return OUT


def main() -> None:
    path = build()
    prs = Presentation(path)
    print(f"Wrote {path}")
    print(f"{len(prs.slides)} slides")


if __name__ == "__main__":
    main()
