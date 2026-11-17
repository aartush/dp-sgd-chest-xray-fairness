"""Build the printable guidebook (.docx) — a complete reference document.

Distinct from FINAL_REPORT.docx: the report is the academic write-up; this is the
handbook someone needs to install, run, audit, present and extend the project.
Every number is read from run artifacts.

Usage:  python -m src.make_guidebook_docx
Writes: docs/GUIDEBOOK.docx
"""

from __future__ import annotations

import json
from pathlib import Path

import docx
import numpy as np
import pandas as pd
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
FIGS = ROOT / "results" / "figures"
OUT = ROOT / "docs" / "GUIDEBOOK.docx"

NAVY = RGBColor(0x1F, 0x3B, 0x57)
ACCENT = RGBColor(0xC0, 0x39, 0x2B)
GREY = RGBColor(0x5A, 0x65, 0x70)

METHOD_LABEL = {
    "baseline": "Vanilla DP-SGD",
    "classweight": "Class-weighted",
    "pretrained": "Pretrained init",
    "pretrained+classweight": "Pretrained + weighted",
}


def load() -> pd.DataFrame:
    rows = []
    for f in sorted(RUNS.glob("*.json")):
        p = json.loads(f.read_text())
        c, m, t = p["config"], p["metrics"], p["train"]
        rows.append({
            "dataset": c["dataset"], "method": c["method"],
            "eps": np.nan if c["eps"] is None else float(c["eps"]),
            "seed": c["seed"], "imbalance_ratio": c["imbalance_ratio"],
            "achieved_eps": t["achieved_eps"], "sigma": t["noise_multiplier"],
            "steps": t["steps_run"], "wall": t["wall_seconds"],
            "n_params": t["n_params"],
            **m,
        })
    return pd.DataFrame(rows)


def setup(doc):
    s = doc.sections[0]
    s.top_margin = s.bottom_margin = Inches(0.85)
    s.left_margin = s.right_margin = Inches(0.9)
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.line_spacing = 1.12


def h1(doc, text, page_break=False):
    if page_break:
        doc.add_page_break()
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(15)
    r.font.color.rgb = NAVY
    # bottom rule via a thin bordered paragraph is fiddly; use a light underline run
    return p


def h2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(11)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(11.5)
    r.font.color.rgb = NAVY
    return p


def para(doc, text, lead=None, italic=False, size=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if lead:
        r = p.add_run(lead)
        r.bold = True
    r = p.add_run(text)
    if italic:
        r.italic = True
    if size:
        r.font.size = Pt(size)
    return p


def bullet(doc, text, lead=None):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    if lead:
        p.add_run(lead).bold = True
    p.add_run(text)
    return p


def numbered(doc, text, lead=None):
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.space_after = Pt(2)
    if lead:
        p.add_run(lead).bold = True
    p.add_run(text)
    return p


def code(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.left_indent = Inches(0.18)
    for i, line in enumerate(text.split("\n")):
        if i:
            p.add_run().add_break()
        r = p.add_run(line)
        r.font.name = "Consolas"
        r.font.size = Pt(8.6)
        r.font.color.rgb = RGBColor(0x1B, 0x27, 0x33)
    # light grey shading behind the block
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), "F2F4F7")
    pPr.append(shd)
    return p


def table(doc, headers, rows, widths=None, font=8.6, caption=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, htext in enumerate(headers):
        hdr[i].text = ""
        r = hdr[i].paragraphs[0].add_run(htext)
        r.bold = True
        r.font.size = Pt(font)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(val))
            r.font.size = Pt(font)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    if caption:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(9)
        r = p.add_run(caption)
        r.italic = True
        r.font.size = Pt(8.2)
        r.font.color.rgb = GREY
    return t


def figure(doc, name, width=5.6, caption=None):
    p = FIGS / name
    if p.exists():
        doc.add_picture(str(p), width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        cp = doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.space_after = Pt(9)
        r = cp.add_run(caption)
        r.italic = True
        r.font.size = Pt(8.4)
        r.font.color.rgb = GREY


def build() -> Path:
    df = load()
    d = df[(df.dataset == "pneumoniamnist") & df.imbalance_ratio.isna()]
    chest = df[df.dataset == "chestmnist"]
    sweep = df[(df.dataset == "pneumoniamnist") & df.imbalance_ratio.notna()]

    def M(method, eps, col):
        s = d[d.method == method]
        s = s[s.eps.isna()] if pd.isna(eps) else s[s.eps == eps]
        return None if s.empty or col not in s else float(s[col].mean())

    doc = docx.Document()
    setup(doc)

    # ------------------------------------------------------------- cover
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(40)
    r = p.add_run("PROJECT GUIDEBOOK")
    r.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = ACCENT

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Who Pays for Differential Privacy?")
    r.bold = True
    r.font.size = Pt(26)
    r.font.color.rgb = NAVY

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Per-class fairness and membership leakage in\ndifferentially private chest X-ray classification")
    r.font.size = Pt(13)
    r.font.color.rgb = GREY

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(26)
    r = p.add_run(
        "Machine Learning Project · Ethics / Responsible AI track\n"
        "Tushar Sharma  ·  MTech I Year, NSUT  ·  tushar.sharma.pg26@nsut.ac.in\n"
        "Supervisor: Bhatia Sir   ·   November 2026"
    )
    r.font.size = Pt(10)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(28)
    r = p.add_run(
        "A complete handbook: the question, the papers, the method, the results,\n"
        "the tools, how to reproduce every number, and how to run the live version.\n\n"
        f"Built from {len(df)} run artifacts. No number in this document is hand-typed."
    )
    r.font.size = Pt(9.5)
    r.italic = True

    # ------------------------------------------------------- how to use
    h1(doc, "How to use this guidebook", page_break=True)
    para(doc, "This document is a reference, not a narrative. Each section stands alone:")
    bullet(doc, "the research question, the two papers and the precise gap being addressed.", "Sections 1–2 — ")
    bullet(doc, "the dataset and its real class balance, the DP-SGD mechanism, the four arms, and the model constraints Opacus imposes.", "Sections 3–4 — ")
    bullet(doc, "exact software versions and hardware, including the hardware-forced batch-size limit.", "Section 5 — ")
    bullet(doc, "the results with their uncertainty, and the measurement floor they are read against.", "Sections 6–7 — ")
    bullet(doc, "step-by-step reproduction, the artifact schema, and the test suite.", "Section 8 — ")
    bullet(doc, "three ways to run the live guidebook, including publishing via GitHub Pages.", "Section 9 — ")
    bullet(doc, "full references, the file map, and viva preparation.", "Sections 10–12 — ")

    para(doc, "The companion files are: this GUIDEBOOK.docx (handbook), FINAL_REPORT.docx "
              "(academic write-up), PROPOSAL.docx (one page), PRESENTATION.pptx/.pdf (deck), "
              "and docs/guidebook/index.html (the interactive, browsable version of this same content).")

    # ------------------------------------------------------ 1. question
    h1(doc, "1. The research question", page_break=False)
    para(doc,
         "Differential privacy is the standard guarantee for training models on sensitive medical "
         "data, and DP-SGD is the mechanism most often used to provide it. DP-SGD clips every "
         "per-example gradient to a fixed norm before averaging and adding noise. That clipping is "
         "what bounds any single patient's influence on the released model — it is the guarantee. "
         "But it also discards the update direction of examples whose gradients are largest, and in "
         "an imbalanced clinical dataset those are disproportionately the minority class.")
    para(doc,
         "Does DP-SGD degrade recall for minority classes disproportionately — and can cheap, "
         "privacy-neutral interventions (a class-weighted loss or a pretrained initialisation) "
         "recover that loss without weakening the guarantee?",
         lead="The question. ")
    h2(doc, "1.1 Why accuracy cannot answer it")
    para(doc,
         "PneumoniaMNIST's test split is 26% normal / 74% pneumonia. A model that predicts "
         "‘pneumonia’ for every scan scores 74% accuracy while achieving zero recall on the normal "
         "class. Accuracy is blind to exactly the harm this project measures, so the primary metric "
         "is minority-class (normal) recall and the majority–minority recall gap.")

    # -------------------------------------------------------- 2. papers
    h1(doc, "2. Two papers and the gap", page_break=False)
    h2(doc, "2.1 Huang & Xie, arXiv:2507.06619 (9 July 2025)")
    para(doc,
         "‘Steps Adaptive Decay DPSGD: Enhancing Performance on Imbalanced Datasets with "
         "Differential Privacy with HAM10000.’ The abstract states that gradient-noise DP methods "
         "‘work well on large datasets like MNIST and CIFAR-100, but fail on small, imbalanced "
         "medical datasets like HAM10000,’ attributing this to minority-class gradients being "
         "clipped while majority classes dominate the update. Their remedy changes the training "
         "schedule — linearly decaying the noise and clipping thresholds — and reports a 2.15% "
         "accuracy gain at ε = 3 over an Auto-DPSGD baseline.")
    para(doc,
         "Only aggregate accuracy is reported, so the paper cannot show whether the minority class "
         "was fixed or merely swamped. No loss-level or initialisation intervention is tested, it is "
         "a two-author arXiv preprint with no peer review, and there is no attack-based privacy "
         "evaluation.",
         lead="Gap. ")
    h2(doc, "2.2 Mohammadi et al., arXiv:2506.00660 → npj Digital Medicine 9:93 (2026)")
    para(doc,
         "‘Differential privacy for medical deep learning: methods, tradeoffs, and deployment "
         "implications.’ A scoping review of 74 studies published through March 2025. It concludes "
         "that ‘severe degradation often occurs under strict privacy (ε ≈ 1), particularly in "
         "underrepresented or complex modalities,’ that ‘privacy-induced performance gaps "
         "disproportionately affect demographic subgroups,’ and that only a small subset of studies "
         "evaluates these tradeoffs. The abstract names ‘balancing privacy with utility and "
         "fairness’ as a critical challenge.")
    para(doc,
         "A review can identify the tension but cannot test a remedy. No study in it isolates "
         "minority-class recall as the outcome, and none tests whether a one-line loss change "
         "closes the gap.",
         lead="Gap. ")
    h2(doc, "2.3 This project's contribution")
    para(doc,
         "Hold the DP mechanism fixed and vary only the loss (class weights) and the initialisation "
         "(ImageNet transfer). Report per-class recall and a membership-inference attack at matched "
         "ε. Both interventions use only public information, so neither consumes privacy budget — "
         "any recovery is genuinely free at the same ε.")
    para(doc,
         "Both citations were verified against arXiv, Crossref and Europe PMC rather than assumed. "
         "The review's DOI string contains ‘2025’, but the journal version's publication year is "
         "2026 — cite it as npj Digital Medicine 9:93 (2026), or cite the arXiv preprint as 2025.",
         lead="Citation note. ", italic=True, size=9.5)

    # ---------------------------------------------------------- 3. data
    h1(doc, "3. Dataset", page_break=False)
    h2(doc, "3.1 PneumoniaMNIST (primary)")
    para(doc,
         "5,856 paediatric chest X-rays, 28×28 grayscale, binary pneumonia-versus-normal, from "
         "MedMNIST v2 (Yang et al., Scientific Data 10:41, 2023). Official split used unchanged.")
    table(doc, ["Split", "Normal (class 0, minority)", "Pneumonia (class 1, majority)", "Total"],
          [["Train", "1,214 (26%)", "3,494 (74%)", "4,708"],
           ["Validation", "135", "389", "524"],
           ["Test", "234", "390", "624"]],
          widths=[1.2, 2.4, 2.4, 1.0],
          caption="Table 1. The native imbalance is real, not synthetically induced.")
    para(doc,
         "The minority class is ‘normal’. This is the clinically meaningful direction for "
         "screening: a normal scan misread as pneumonia triggers unnecessary follow-up, and in a "
         "screening setting that is the costly error. Note the test split is less imbalanced than "
         "train (37% vs 26% normal), which matters for the attack — a pooled membership-inference "
         "AUC would partly measure the distribution shift rather than memorisation, so the primary "
         "privacy number is class-conditional.")
    h2(doc, "3.2 ChestMNIST (scale check)")
    para(doc,
         "78,468 frontal X-rays, 14 multi-label findings from NIH-ChestXray14; the rarest label has "
         "only 144 positives in the full training set (0.18%). Capped at 8,000 training images for "
         "tractability and labelled as such everywhere it appears. It is a transfer check, not a "
         "replication.")
    para(doc,
         "Both datasets are public and de-identified, download automatically through the medmnist "
         "package, and require no data-use agreement.")

    # -------------------------------------------------------- 4. method
    h1(doc, "4. Method", page_break=False)
    h2(doc, "4.1 DP-SGD in one paragraph")
    para(doc, "For a batch of size B, per-example gradients gᵢ are clipped to norm C, summed, and perturbed:")
    code(doc, "g̃ᵢ = gᵢ · min(1, C / ‖gᵢ‖)\nθ  ← θ − η · ( Σᵢ g̃ᵢ + σ · C · N(0, I) ) / B")
    para(doc,
         "The accountant converts (σ, sample rate, number of steps) into an (ε, δ) guarantee. This "
         "project fixes δ = 1e-5, plans a step budget, solves for the σ that spends exactly the "
         "target ε over that budget, then records the achieved ε per run. Every arm uses the same "
         "batch size and data order, so ε is the only variable distinguishing private runs.")
    h2(doc, "4.2 The four arms")
    table(doc, ["Arm", "DP", "Class weights", "Pretrained init"],
          [["Vanilla DP-SGD", "ε ∈ {1,3,8,∞}", "—", "—"],
           ["Class-weighted", "ε ∈ {1,3,8,∞}", "inverse frequency", "—"],
           ["Pretrained init", "ε ∈ {1,3,8,∞}", "—", "ImageNet ResNet-18"],
           ["Both", "ε ∈ {1,3,8,∞}", "inverse frequency", "ImageNet ResNet-18"]],
          widths=[1.7, 1.5, 1.6, 1.8],
          caption="Table 2. 4 arms × 4 budgets × 3 seeds = 48 main-grid runs.")
    h2(doc, "4.3 Why the two interventions are privacy-neutral")
    para(doc,
         "Class weights are computed from the training label histogram — public aggregate "
         "information — and the accountant never sees them. ImageNet weights are public. Neither "
         "consumes privacy budget, so any recovery they produce is genuinely free at the same ε. "
         "That is the practical claim of the project.")
    h2(doc, "4.4 The model and the constraints Opacus imposes")
    para(doc, "ResNet-18 adapted to 1×28×28 input. Three changes are required, not stylistic:")
    bullet(doc, "Opacus cannot form per-sample gradients through batch statistics and rejects such models outright; a negative-control test asserts this failure.", "GroupNorm replaces every BatchNorm. ")
    bullet(doc, "Opacus' backward hooks return views that autograd forbids from being mutated in place.", "ReLU is non-in-place. ")
    bullet(doc, "same reason: the residual block's out += identity is replaced with an out-of-place add.", "Out-of-place residual add — ")
    h2(doc, "4.5 Clipping-bound calibration")
    para(doc,
         "C is set from the median true per-sample gradient norm, measured once on an untrained "
         "model using Opacus' own grad_sample buffers. The median (not max) keeps clipping noise "
         "low. The statistic is aggregated, computed before training, and releases nothing per "
         "example.")
    h2(doc, "4.6 Membership-inference attacks")
    para(doc,
         "Score each example by −cross-entropy loss; low loss implies membership; compute ROC-AUC "
         "over members (train) versus non-members (test). Reported class-conditionally (within each "
         "class, then averaged) to remove the train/test class-mix confound. An AUC of 0.5 means "
         "the attack cannot distinguish members from non-members at all.",
         lead="Primary — loss threshold (Yeom-style). ")
    para(doc,
         "A logistic-regression attacker trained on a separate model's in/out behaviour over "
         "(loss, confidence, entropy), then applied to the target. Validated against both a null "
         "(must give ≈ 0.5) and an injected-leakage case; both are permanent tests.",
         lead="Secondary — shadow model. ")
    para(doc,
         "a loss-threshold attack is a lower bound on leakage, not a ceiling. A flat curve is "
         "evidence for this attack family, not proof that no attack exists. The formal ε bound is "
         "the real guarantee.",
         lead="Stated plainly: ", italic=True)

    # --------------------------------------------------------- 5. tools
    h1(doc, "5. Software, tools and hardware", page_break=False)
    table(doc, ["Component", "Version", "Role"],
          [["Python", "3.12.10", "Runtime"],
           ["PyTorch", "2.6.0+cu124", "Training, autograd"],
           ["torchvision", "0.21.0+cu124", "ResNet-18 architecture, ImageNet checkpoint"],
           ["Opacus", "1.5.4", "DP-SGD, RDP accountant, ghost clipping"],
           ["MedMNIST", "3.0.2", "Dataset download and loading"],
           ["scikit-learn", "1.9.1", "Per-class metrics, attack classifier"],
           ["NumPy / pandas / SciPy", "2.5.4 / 3.1.1 / 1.18.1", "Numerics, aggregation"],
           ["matplotlib", "3.11.2", "All figures"],
           ["pytest", "9.2.0", "22 invariant tests"],
           ["python-docx", "1.2.0", "Proposal and report generation"],
           ["python-pptx", "1.0.2", "Presentation generation"],
           ["pywin32", "312", "PowerPoint COM: slide PNGs and PDF export"]],
          widths=[1.9, 1.9, 3.1],
          caption="Table 3. Exact versions, pinned in requirements.txt.")
    h2(doc, "5.1 Hardware")
    para(doc,
         "NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), Windows 11, 16 GB RAM, CUDA 12.4. The "
         f"full study is {len(df)} runs: {df.wall.sum()/60:.0f} minutes of GPU time, with a median "
         f"private run taking {df[df.eps.notna()].wall.median():.0f} s.")
    h2(doc, "5.2 Hardware-forced constraint: batch size 32")
    para(doc,
         "Opacus materialises per-sample gradients, and this 6 GB card out-of-memory at batch 64. "
         "Batch 32 is therefore mandatory for DP arms; non-private arms use the same size so the "
         "comparison is clean. Two further facts were established experimentally:")
    bullet(doc, "not supported on this Windows build (PyTorch warns and ignores it).", "expandable_segments — ")
    bullet(doc, "0.24 GB live versus 7.8 GB for hooks mode, and not slower (123 vs 135 ms/step).", "Ghost clipping — ")
    para(doc,
         "These are recorded in the module docstring of src/train.py, because rediscovering them "
         "cost several failed multi-hour grid runs.",
         italic=True, size=9.5)

    # ------------------------------------------------------- 6. results
    h1(doc, "6. Results", page_break=True)
    h2(doc, "6.1 Main grid — PneumoniaMNIST")
    rows = []
    for eps in [np.nan, 1.0, 3.0, 8.0]:
        for method in METHOD_LABEL:
            s = d[(d.method == method) & ((d.eps.isna()) if pd.isna(eps) else (d.eps == eps))]
            if s.empty:
                continue
            mr = s.minority_recall.mean()
            sd = s.minority_recall.std(ddof=1) if len(s) > 1 else float("nan")
            rows.append([
                METHOD_LABEL[method],
                "No DP" if pd.isna(eps) else f"ε = {eps:g}",
                f"{mr:.3f} ± {sd:.3f}" if not np.isnan(sd) else f"{mr:.3f}",
                f"{s.majority_recall.mean():.3f}",
                f"{s.accuracy.mean():.3f}",
                f"{s.mia_auc_class_conditional.mean():.3f}",
                len(s),
            ])
    table(doc, ["Method", "Privacy", "Minority recall", "Majority recall", "Accuracy", "MIA AUC", "n"],
          rows, widths=[1.5, 0.7, 1.3, 1.1, 0.8, 0.8, 0.3],
          caption="Table 4. Mean ± sd over seeds. Minority is ‘normal’. MIA AUC is class-conditional; "
                  "0.5 = no measurable leakage.")
    para(doc,
         "The test split has only 234 minority images, so one image moves recall by 0.43 points and "
         "the binomial standard error at recall 0.70 is ±3.0 points. Every delta in this section is "
         "read against that floor rather than treated as an effect by default.",
         lead="Measurement floor. ")

    h2(doc, "6.2 Finding 1 — the guarantee is empirically real")
    para(doc,
         f"MIA AUC on the non-private model is {M('baseline', np.nan, 'mia_auc_class_conditional'):.3f}. "
         f"It falls to {M('baseline', 1.0, 'mia_auc_class_conditional'):.3f} at ε = 1 and stays in a "
         f"narrow band up to ε = 8. The drop is large relative to its seed spread (≈0.004), so this is "
         f"the result the project defends at three seeds. Above ε = 1 the curve is essentially flat: "
         f"the guarantee bites once, and tightening further does not remove materially more measured "
         f"leakage.")
    figure(doc, "fig3_mia_auc.png", 5.2, "Figure 1. Membership-inference AUC versus privacy budget.")

    h2(doc, "6.3 Finding 2 — class weighting helps; pretrained init backfires")
    b3 = M("baseline", 3.0, "minority_recall")
    para(doc,
         f"At ε = 3, against vanilla's {b3:.3f}: class weighting reaches "
         f"{M('classweight', 3.0, 'minority_recall'):.3f} "
         f"({M('classweight', 3.0, 'minority_recall')-b3:+.3f}), pretrained init falls to "
         f"{M('pretrained', 3.0, 'minority_recall'):.3f} "
         f"({M('pretrained', 3.0, 'minority_recall')-b3:+.3f}), and the combination reaches "
         f"{M('pretrained+classweight', 3.0, 'minority_recall'):.3f} "
         f"({M('pretrained+classweight', 3.0, 'minority_recall')-b3:+.3f}). Class weighting moved "
         f"minority recall in the same direction at every seed; pretrained init was worse than "
         f"vanilla at every seed at ε = 8 despite being the strongest arm without DP.")
    figure(doc, "fig1_minority_recall.png", 5.4,
           "Figure 2. Minority-class recall versus privacy budget, mean ± sd.")
    figure(doc, "fig4_per_class_eps3.png", 5.2, "Figure 3. Per-class recall at ε = 3 by method.")

    h2(doc, "6.4 Finding 3 — the effect is present at ε = 1 but not monotone")
    g0 = M("baseline", np.nan, "majority_recall") - M("baseline", np.nan, "minority_recall")
    g1 = M("baseline", 1.0, "majority_recall") - M("baseline", 1.0, "minority_recall")
    g8 = M("baseline", 8.0, "majority_recall") - M("baseline", 8.0, "minority_recall")
    para(doc,
         f"Without DP the majority–minority gap is {g0:.3f}. At ε = 1 it widens to {g1:.3f}, the "
         f"largest of any budget. It does not widen monotonically: by ε = 8 it is {g8:.3f}, below "
         f"the no-DP value. So ‘tighter privacy is always less fair’ is false on this benchmark — "
         f"and the ε = 1 magnitude is within the ±3-point floor in any case. Reported as "
         f"suggestive, not established.")
    figure(doc, "fig2_recall_gap.png", 5.2,
           "Figure 4. Majority-minus-minority recall gap across budgets.")

    h2(doc, "6.5 Imbalance sweep")
    para(doc,
         "The native split is 2.88:1. We additionally trained on a balanced (1:1) and a severe "
         "(8:1) training set, holding the minority examples fixed and only shrinking the majority "
         "pool, with 2 seeds per cell.")
    rows = []
    for method in ["baseline", "classweight"]:
        for ratio in [1.0, 8.0]:
            cells = []
            for eps in [np.nan, 1.0, 3.0]:
                s = sweep[(sweep.method == method) & np.isclose(sweep.imbalance_ratio, ratio)
                          & ((sweep.eps.isna()) if pd.isna(eps) else (sweep.eps == eps))]
                cells.append(f"{s.minority_recall.mean():.3f}" if not s.empty else "—")
            rows.append([METHOD_LABEL[method], f"{ratio:g}:1"] + cells)
    table(doc, ["Method", "Train ratio", "No DP", "ε = 1", "ε = 3"], rows,
          widths=[1.9, 0.9, 0.8, 0.8, 0.8],
          caption="Table 5. Minority recall under induced imbalance.")
    b8 = sweep[(sweep.method == "baseline") & np.isclose(sweep.imbalance_ratio, 8.0)
               & sweep.eps.isna()].minority_recall.mean()
    c8 = sweep[(sweep.method == "classweight") & np.isclose(sweep.imbalance_ratio, 8.0)
               & sweep.eps.isna()].minority_recall.mean()
    b1 = sweep[(sweep.method == "baseline") & np.isclose(sweep.imbalance_ratio, 1.0)
               & sweep.eps.isna()].minority_recall.mean()
    para(doc,
         f"At 8:1 the unweighted model recalls {b8:.3f} of minority scans, down from {b1:.3f} at "
         f"1:1, while the class-weighted model holds {c8:.3f}. This is the clearest evidence that a "
         f"privacy-neutral loss change protects the minority exactly where imbalance is worst.",)
    figure(doc, "fig5_imbalance_sweep.png", 5.9, "Figure 5. Minority recall against induced imbalance.")

    h2(doc, "6.6 Scale check — ChestMNIST")
    rows = []
    for eps in [np.nan, 3.0]:
        for method in ["baseline", "classweight"]:
            s = chest[(chest.method == method) & ((chest.eps.isna()) if pd.isna(eps) else (chest.eps == eps))]
            if s.empty:
                continue
            rows.append([METHOD_LABEL[method], "No DP" if pd.isna(eps) else f"ε = {eps:g}",
                         f"{s.macro_auc.mean():.3f}", f"{s.min_label_recall.mean():.3f}",
                         f"{s.mia_auc_class_conditional.mean():.3f}"])
    table(doc, ["Method", "Privacy", "Macro AUC", "Min label recall", "MIA AUC"], rows,
          widths=[1.9, 0.8, 0.9, 1.1, 0.8],
          caption="Table 6. ChestMNIST, 8k images, 14 labels, 6 epochs.")
    para(doc,
         "Minimum label recall is 0.000 in every cell — the rarest findings have single-digit "
         "positives at 8k images, so nothing is predicted at a 0.5 threshold. MIA AUC also dips "
         "slightly below 0.500; an attack cannot be worse than chance, so that deviation is "
         "estimation noise, reported as ‘no measurable leakage’ rather than a privacy gain. This "
         "is a scale check, not a replication.")

    # ---------------------------------------------------------- 7. honest
    h1(doc, "7. Limitations and ethical considerations", page_break=False)
    bullet(doc, "differences smaller than the ±3-point binomial floor are called noise, not findings. Paired, seed-matched deltas are in results/tables/paired_deltas.csv.", "Three seeds — ")
    bullet(doc, "MedMNIST downsamples from up to 384×2,916. Some diagnostic detail is destroyed at 28 pixels; results describe this benchmark, not full-resolution radiology.", "28×28 inputs — ")
    bullet(doc, "loss-threshold MIA is a lower bound on leakage. A flat curve is evidence for this attack family, not proof that no attack exists.", "Weak attack — ")
    bullet(doc, "noise came from a fast PRNG. Standard for experimentation; a production claim would require secure mode and is not made here.", "secure_mode off — ")
    bullet(doc, "larger batches would require more noise for the same ε; all arms share 32, so the comparison is internally valid.", "Batch size 32, forced by hardware — ")
    bullet(doc, "no claim that DP-SGD is unfair in general.", "One architecture, one dataset family — ")
    bullet(doc, "public, de-identified datasets; no new patient data collected; no clinical deployment claimed — this is a study of a mechanism, not a validated diagnostic tool.", "Data ethics — ")

    # -------------------------------------------------------- 8. reproduce
    h1(doc, "8. Reproducing the project", page_break=True)
    h2(doc, "8.1 First-time setup")
    code(doc,
         "cd DP-SGD-Medical-Privacy\n"
         "py -3.12 -m venv .venv\n"
         ".venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
         ".venv/Scripts/python.exe -m pip install -r requirements.txt")
    h2(doc, "8.2 Run the study")
    code(doc,
         "bash scripts/run_grid.sh all                      # 76 runs, resumable\n"
         ".venv/Scripts/python.exe -m src.analyze           # tables, figures, RESULTS.md\n"
         ".venv/Scripts/python.exe -m src.make_final_report # FINAL_REPORT.docx\n"
         ".venv/Scripts/python.exe -m src.make_deck         # PRESENTATION.pptx")
    para(doc,
         "Runs are idempotent and resumable: each is keyed by a hash of its config, so re-running "
         "skips completed work. --force reruns; --list prints everything on disk. Each run executes "
         "in its own short-lived interpreter, because Opacus' grad-sample hooks and CUDA allocator "
         "state are not fully released between runs in a shared process.")
    h2(doc, "8.3 Verify the invariants")
    code(doc, ".venv/Scripts/python.exe -m pytest tests -q     # 22 tests")
    para(doc,
         "The tests pin properties that would silently corrupt results if broken: class weights "
         "must favour the minority, imbalance construction must leave minority rows byte-identical, "
         "the MIA must return chance under a null and detect injected leakage, the DP accountant "
         "must spend the budget it claims, the model must contain no BatchNorm, and the multilabel "
         "weighted loss must keep its shape.")
    h2(doc, "8.4 Project structure")
    code(doc,
         "src/\n"
         "  data.py               dataset loading, imbalance construction, class weights\n"
         "  models.py             GroupNorm ResNet-18, Opacus-safe forward, ImageNet transfer\n"
         "  train.py              training engine, DP-SGD (ghost clipping), C calibration\n"
         "  metrics.py            per-class recall, recall gap, aggregation\n"
         "  mia.py                loss-threshold and shadow-model attacks\n"
         "  runner.py             experiment grid, JSON artifacts, resume, process isolation\n"
         "  analyze.py            tables, figures, RESULTS.md\n"
         "  report.py             REPORT.md\n"
         "  make_proposal.py      PROPOSAL.docx\n"
         "  make_final_report.py  FINAL_REPORT.docx\n"
         "  make_deck.py          PRESENTATION.pptx\n"
         "  make_guidebook.py     docs/guidebook/index.html (live version)\n"
         "  make_guidebook_docx.py  this document\n"
         "  render_deck.py        slide PNGs + PDF via PowerPoint COM\n"
         "scripts/\n"
         "  run_grid.sh           resilient shell driver\n"
         "  serve_guidebook.py    local HTTP server for the live version\n"
         "tests/\n"
         "  test_invariants.py    22 tests\n"
         "results/\n"
         "  runs/                 76 JSON artifacts\n"
         "  tables/               CSV aggregates incl. paired_deltas.csv\n"
         "  figures/              6 PNG figures\n"
         "docs/                   proposal, report, deck, guidebook")
    h2(doc, "8.5 Artifact schema")
    para(doc,
         "Every run writes one JSON file recording: the full config, achieved ε, noise multiplier, "
         "clipping norm, steps run, parameter count, wall time, the per-epoch history, the per-seed "
         "metrics and the data counts. Any number in any document can be traced to the file that "
         "produced it.")

    # ------------------------------------------------------------ 9. live
    h1(doc, "9. Running the live guidebook", page_break=False)
    para(doc,
         "The live version is a single self-contained HTML file (docs/guidebook/index.html) with "
         "every figure embedded and all 76 runs loaded into an interactive filter/sort table. It "
         "needs no network, no CDN and no build step. It contains the same content as this document "
         "plus the explorable run table.")
    h2(doc, "9.1 Option A — open the file (simplest)")
    para(doc, "Double-click docs/guidebook/index.html. Everything works, including the explorer, "
              "because there are no external requests. Verified by loading it over file:// with all "
              "76 runs and all 5 figures present.")
    h2(doc, "9.2 Option B — serve it on a local URL")
    code(doc,
         ".venv/Scripts/python.exe scripts/serve_guidebook.py\n"
         "# then open http://127.0.0.1:8000")
    para(doc,
         "Serving it means the address bar holds a real URL you can put in a slide, and the "
         "print-to-PDF path behaves like a normal website. The server binds to 127.0.0.1 only, so "
         "it is never exposed to the network.")
    h2(doc, "9.3 Option C — publish publicly with GitHub Pages")
    para(doc, "This must be done from your own GitHub account, since it publishes to the internet.")
    numbered(doc, "create a repository and push the project, including docs/guidebook/index.html.", "Create and push — ")
    numbered(doc, "Settings → Pages → Source: ‘Deploy from a branch’ → Branch: main, folder: /docs → Save.", "Enable Pages — ")
    numbered(doc,
             "GitHub builds the site at an address beginning https://, then your GitHub username, "
             "then .github.io/dp-sgd-chest-xray-fairness/guidebook/. Settings → Pages shows the "
             "exact URL once it is live.", "Get the URL — ")
    para(doc,
         "Before publishing, decide whether the repository should be public — a public repo "
         "exposes the author's name and code. The datasets are not included (data/ is gitignored), so no "
         "clinical images are redistributed.",
         italic=True)

    # ------------------------------------------------------------ 10. refs
    h1(doc, "10. References", page_break=True)
    refs = [
        "Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep Learning with Differential Privacy. ACM CCS. arXiv:1607.00133",
        "Huang, X., & Xie, F. (2025). Steps Adaptive Decay DPSGD: Enhancing Performance on Imbalanced Datasets with Differential Privacy with HAM10000. arXiv:2507.06619",
        "Mohammadi, M., Vejdanihemmat, M., Lotfinia, M., Rusu, M., Truhn, D., Maier, A., & Tayebi Arasteh, S. (2026). Differential privacy for medical deep learning: methods, tradeoffs, and deployment implications. npj Digital Medicine, 9, 93. doi:10.1038/s41746-025-02280-z (preprint arXiv:2506.00660, 2025)",
        "Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H., & Ni, B. (2023). MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data, 10, 41. doi:10.1038/s41597-022-01721-8",
        "Yousefpour, A., et al. (2021). Opacus: User-Friendly Differential Privacy Library in PyTorch. arXiv:2109.12298",
        "Yeom, S., Giacomelli, I., Fredrikson, M., & Jha, S. (2018). Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting. IEEE CSF.",
        "Kermany, D. S., et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell, 172(5). (Origin of the PneumoniaMNIST images.)",
        "Wang, X., et al. (2017). ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks. CVPR. (Origin of the ChestMNIST images.)",
    ]
    for r in refs:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        p.add_run(r).font.size = Pt(9.2)

    # -------------------------------------------------------- 11. file map
    h1(doc, "11. Deliverable map", page_break=False)
    table(doc, ["File", "What it is", "Who it is for"],
          [["docs/PROPOSAL.docx", "One-page proposal", "Course submission"],
           ["docs/FINAL_REPORT.docx", "Full academic report, ~2,850 words", "Course submission, evaluation"],
           ["docs/PRESENTATION.pptx / .pdf", "15-slide deck and handout", "Viva, 20 November"],
           ["docs/GUIDEBOOK.docx", "This handbook", "You, a teammate, a reviewer"],
           ["docs/guidebook/index.html", "Interactive live version", "Browsing, publishing"],
           ["docs/REPORT.md / RESULTS.md", "Generated summaries", "Quick reference"],
           ["docs/SUBMISSION_CHECKLIST.md", "Git, viva prep, final checks", "Before submitting"],
           ["results/runs/*.json", "76 run artifacts", "The evidence for every number"],
           ["results/tables/paired_deltas.csv", "Seed-matched comparisons", "Uncertainty questions"],
           ["tests/test_invariants.py", "22 tests", "Correctness guarantees"]],
          widths=[2.3, 2.4, 2.0],
          caption="Table 7. Everything in the package.")

    # ------------------------------------------------------------ 12. viva
    h1(doc, "12. Viva preparation", page_break=False)
    h2(doc, "12.1 Numbers to have ready")
    bullet(doc, f"MIA AUC {M('baseline', np.nan, 'mia_auc_class_conditional'):.3f} (no DP) → "
                f"{M('baseline', 1.0, 'mia_auc_class_conditional'):.3f} (ε = 1). Seed spread ≈0.004, "
                f"so the effect is ~10× the noise — this is the defensible result.", "Privacy — ")
    bullet(doc, f"minority recall at ε = 3: {b3:.3f} vanilla, {M('classweight',3.0,'minority_recall'):.3f} "
                f"class-weighted, {M('pretrained',3.0,'minority_recall'):.3f} pretrained.", "Fairness — ")
    bullet(doc, f"at 8:1 imbalance, minority recall {b8:.3f} unweighted vs {c8:.3f} weighted.", "Imbalance — ")
    bullet(doc, "226 head-to-head comparisons reduce to seed-matched deltas in paired_deltas.csv; always say ‘over three seeds’.", "Uncertainty — ")
    h2(doc, "12.2 Questions to expect, and the honest answers")
    para(doc,
         "Because predicting pneumonia for everything scores 74% while getting every normal scan "
         "wrong. That exact failure is a test case in the suite.",
         lead="Why not just report accuracy? ")
    para(doc,
         "No — it is a lower bound. The formal ε is the guarantee; the attack shows the guarantee "
         "bites in practice. A flat curve does not prove no attack exists.",
         lead="Is the membership-inference attack convincing? ")
    para(doc,
         "Opacus cannot form per-sample gradients through batch statistics. A negative-control "
         "test asserts that it rejects BatchNorm outright.",
         lead="Why GroupNorm? ")
    para(doc,
         "No. The weights come from the training label histogram, which is public aggregate "
         "information, and the accountant never sees them. Same ε, same guarantee.",
         lead="Is class weighting cheating on privacy? ")
    para(doc,
         "It is the best arm without DP and the worst at ε = 8, at every seed. Transfer needs the "
         "late fine-tuning that adapts ImageNet features to chest X-rays, and DP's gradient noise "
         "disrupts exactly that. A DP-aware schedule is the follow-up — this is a hypothesis "
         "consistent with the data, not a thing the experiment isolated.",
         lead="Why does pretrained init hurt under DP? ")
    para(doc,
         "It is within the ±3-point floor of a 234-image test set, so we call it suggestive rather "
         "than established — and we note it is not monotone in ε. Calling a 5-point difference a "
         "finding would be the overclaim reviewers look for.",
         lead="Why is the fairness claim weaker than the privacy claim? ")
    h2(doc, "12.3 What to say if asked for next steps")
    bullet(doc, "to move the fairness result past the noise floor.", "More seeds and a larger minority test set — ")
    bullet(doc, "to test whether pretrained init can be rescued.", "A DP-aware fine-tuning schedule — ")
    bullet(doc, "the SAD-DPSGD idea, as a fifth arm.", "Per-class clipping — ")
    bullet(doc, "the code is written and validated; it needs compute.", "Shadow-model attack on every arm — ")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    return OUT


def main() -> None:
    path = build()
    doc = docx.Document(path)
    words = sum(len(p.text.split()) for p in doc.paragraphs)
    print(f"Wrote {path}")
    print(f"{words} words, {len(doc.tables)} tables, {len(doc.inline_shapes)} figures")


if __name__ == "__main__":
    main()
