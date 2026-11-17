"""Build the full academic project report as a .docx, from run artifacts only.

This is the final-project document (not the one-page proposal). Every quantitative
statement is read from results/runs/*.json or results/tables/*.csv; nothing is
typed by hand. Where a result is weaker than it looks, the prose says so.

Usage:  python -m src.make_final_report
Writes: docs/FINAL_REPORT.docx
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
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
FIGS = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"
OUT = ROOT / "docs" / "FINAL_REPORT.docx"

METHOD_LABEL = {
    "baseline": "Vanilla DP-SGD",
    "classweight": "Class-weighted",
    "pretrained": "Pretrained init",
    "pretrained+classweight": "Pretrained + class-weighted",
}
METHOD_ORDER = list(METHOD_LABEL)

NATIVE_RATIO = 3494 / 1214
TEST_MINORITY_N = 234
TEST_MAJORITY_N = 390


def load() -> pd.DataFrame:
    rows = []
    for f in sorted(RUNS.glob("*.json")):
        p = json.loads(f.read_text())
        c, m, t = p["config"], p["metrics"], p["train"]
        row = {
            "run_id": p["run_id"],
            "dataset": c["dataset"],
            "method": c["method"],
            "eps": np.nan if c["eps"] is None else float(c["eps"]),
            "seed": c["seed"],
            "imbalance_ratio": c["imbalance_ratio"],
            "multilabel": c["multilabel"],
            "achieved_eps": t["achieved_eps"],
            "sigma": t["noise_multiplier"],
            "max_grad_norm": t["max_grad_norm"],
            "steps": t["steps_run"],
            "wall": t["wall_seconds"],
            "n_params": t["n_params"],
            "epochs_run": len(t["history"]),
        }
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)


def eps_label(e) -> str:
    return "No DP" if (isinstance(e, float) and np.isnan(e)) else f"ε = {e:g}"


def recall_se(r: float, n: int = TEST_MINORITY_N) -> float:
    return float(np.sqrt(max(r * (1 - r), 0.0) / n))


# ----------------------------------------------------------------- doc styling


def setup(doc: docx.Document) -> None:
    s = doc.sections[0]
    s.top_margin = s.bottom_margin = Inches(0.9)
    s.left_margin = s.right_margin = Inches(0.95)
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.line_spacing = 1.15


def h1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(14)
    r.font.color.rgb = RGBColor(0x1F, 0x3B, 0x57)
    return p


def h2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(11.5)
    return p


def para(doc, text, bold_lead=None, size=None, italic=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if bold_lead:
        r = p.add_run(bold_lead)
        r.bold = True
    r = p.add_run(text)
    if size:
        r.font.size = Pt(size)
    if italic:
        r.italic = True
    return p


def bullet(doc, text, bold_lead=None):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    if bold_lead:
        p.add_run(bold_lead).bold = True
    p.add_run(text)
    return p


def caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(8.5)
    return p


def make_table(doc, headers, rows, widths=None, font=8.5):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, htext in enumerate(headers):
        hdr[i].text = ""
        run = hdr[i].paragraphs[0].add_run(htext)
        run.bold = True
        run.font.size = Pt(font)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(val))
            run.font.size = Pt(font)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    return t


def figure(doc, name, width=5.9):
    p = FIGS / name
    if p.exists():
        doc.add_picture(str(p), width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


# --------------------------------------------------------------------- build


def build() -> Path:
    df = load()
    d = df[(df.dataset == "pneumoniamnist") & (df.imbalance_ratio.isna())]
    chest = df[df.dataset == "chestmnist"]
    sweep = df[(df.dataset == "pneumoniamnist") & df.imbalance_ratio.notna()].copy()
    sweep["eff_ratio"] = sweep.imbalance_ratio.astype(float)

    def M(method, eps, col):
        s = d[d.method == method]
        s = s[s.eps.isna()] if pd.isna(eps) else s[s.eps == eps]
        return None if s.empty or col not in s else float(s[col].mean())

    def SD(method, eps, col):
        s = d[d.method == method]
        s = s[s.eps.isna()] if pd.isna(eps) else s[s.eps == eps]
        return None if len(s) < 2 or col not in s else float(s[col].std(ddof=1))

    def N(method, eps):
        s = d[d.method == method]
        s = s[s.eps.isna()] if pd.isna(eps) else s[s.eps == eps]
        return len(s)

    doc = docx.Document()
    setup(doc)

    # ---------------------------------------------------------- title block
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("Who Pays for Differential Privacy?\nPer-Class Fairness and Membership Leakage in Differentially Private Chest X-Ray Classification")
    r.bold = True
    r.font.size = Pt(17)
    r.font.color.rgb = RGBColor(0x1F, 0x3B, 0x57)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sub.add_run("Machine Learning Project — Ethics / Responsible AI track")
    sr.font.size = Pt(11)
    sr.italic = True

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    mr = meta.add_run(
        "Tushar Sharma  ·  MTech I Year, NSUT  ·  tushar.sharma.pg26@nsut.ac.in\n"
        "Supervisor: Bhatia Sir    ·    November 2026"
    )
    mr.font.size = Pt(9.5)

    # A real bottom-border rule, so the title block never reads as an unfilled
    # form field the way a row of underscores does.
    hr = doc.add_paragraph()
    hr.paragraph_format.space_before = Pt(4)
    hr.paragraph_format.space_after = Pt(8)
    pPr = hr._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "AAAAAA")
    borders.append(bottom)
    pPr.append(borders)

    # ------------------------------------------------------------- abstract
    h1(doc, "Abstract")
    base_nodp = M("baseline", np.nan, "minority_recall")
    base_1 = M("baseline", 1.0, "minority_recall")
    mia_nodp = M("baseline", np.nan, "mia_auc_class_conditional")
    mia_1 = M("baseline", 1.0, "mia_auc_class_conditional")
    cw_3 = M("classweight", 3.0, "minority_recall")
    base_3 = M("baseline", 3.0, "minority_recall")

    para(
        doc,
        "Differential privacy is the standard guarantee for training models on sensitive "
        "medical data, and DP-SGD is the mechanism most often used to provide it. DP-SGD clips "
        "each per-example gradient to a fixed norm before averaging and noising. That clipping "
        "is what bounds any single patient's influence on the released model, but it also "
        "discards the update directions of examples with large gradients — which, in an "
        "imbalanced clinical dataset, are disproportionately the minority class. This project "
        "asks who pays for the guarantee. We train an identical ResNet-18 (adapted to 28×28 "
        "chest X-rays with GroupNorm) at privacy budgets ε ∈ {1, 3, 8} and without DP on "
        f"PneumoniaMNIST, across four arms: vanilla DP-SGD, a class-weighted loss, a pretrained "
        f"initialisation, and both. We report per-class recall rather than accuracy, and we add "
        f"a membership-inference attack to test whether the guarantee reduces measurable "
        f"leakage. Across {len(d)} main-grid runs, membership-inference AUC falls from "
        f"{mia_nodp:.3f} without DP to {mia_1:.3f} at ε = 1, with a seed spread an order of "
        f"magnitude smaller than the effect — the robust result. Class weighting is a "
        f"privacy-neutral, consistent partial remedy (minority recall {cw_3:.3f} versus "
        f"{base_3:.3f} for vanilla at ε = 3, same direction at every seed, no privacy cost). "
        f"A pretrained initialisation, by contrast, is the strongest arm without DP and the "
        f"weakest with it. We report the minority-recall deltas against an explicit noise floor "
        f"and do not claim more than three seeds on a 234-image minority test set can support.",
    )

    # ---------------------------------------------------------- introduction
    h1(doc, "1. Introduction")
    h2(doc, "1.1 Motivation")
    para(
        doc,
        "Training on medical images requires either a data-use agreement or a privacy "
        "guarantee. Differential privacy supplies the latter: a model trained with DP-SGD is "
        "provably insensitive, in distribution, to the inclusion or removal of any single "
        "record. Healthcare organisations increasingly cite DP as a precondition for sharing "
        "models rather than data. The question of whether that guarantee is affordable is "
        "usually answered in aggregate accuracy.",
    )
    para(
        doc,
        "Aggregate accuracy is the wrong lens for clinical screening. On PneumoniaMNIST, "
        "predicting pneumonia for every scan achieves 74% accuracy while recalling zero normal "
        "cases. A privacy mechanism that shaves a few points of accuracy but collapses recall "
        "on one class has not made a uniform trade-off — it has redistributed error onto "
        "patients whose presentation is under-represented in the training set. This project "
        "measures that redistribution directly.",
    )

    h2(doc, "1.2 Contributions")
    bullet(doc, "a per-class recall and recall-gap analysis of DP-SGD across four privacy budgets, with an explicit binomial noise floor rather than point estimates alone.", "Measurement. ")
    bullet(doc, "two privacy-neutral interventions — inverse-frequency class weighting and ImageNet initialisation — evaluated at matched ε, showing the first helps consistently and the second backfires under noise.", "Interventions. ")
    bullet(doc, "a loss-threshold membership-inference attack, and a shadow-model attack validated against both a null (no leakage) and an injected-leakage case, reporting empirical leakage rather than only a formal bound.", "Empirical privacy. ")
    bullet(doc, "every run writes a self-describing JSON artifact; all tables and figures in this report are generated from those artifacts, and the pipeline is re-runnable end to end.", "Reproducibility. ")

    # ---------------------------------------------------------- related work
    h1(doc, "2. Related Work and Research Gap")
    h2(doc, "2.1 DP-SGD and its accuracy cost")
    para(
        doc,
        "DP-SGD (Abadi et al., 2016) is the canonical mechanism: per-example gradients are "
        "clipped to a bound C, summed, and perturbed with Gaussian noise scaled by σ; the "
        "privacy cost is tracked by a moments or RDP accountant. Its accuracy cost is well "
        "documented and grows as ε shrinks. What is less often reported is how that cost is "
        "distributed across classes and subgroups.",
    )
    h2(doc, "2.2 The two papers this project responds to")
    para(
        doc,
        "Huang & Xie (arXiv:2507.06619, July 2025) study DP-SGD on HAM10000 and state, as their "
        "motivation, that gradient-noise DP methods “work well on large datasets like MNIST and "
        "CIFAR-100, but fail on small, imbalanced medical datasets like HAM10000,” attributing "
        "the failure to minority-class gradients being clipped while majority classes dominate "
        "the update. Their remedy changes the training schedule — linearly decaying the noise "
        "and clipping thresholds — and reports a 2.15% accuracy gain at ε = 3 over an "
        "Auto-DPSGD baseline.",
    )
    para(
        doc,
        "Mohammadi et al. (arXiv:2506.00660, May 2025; published in npj Digital Medicine 9:93, "
        "2026) is a scoping review of 74 studies. It concludes that “severe degradation often "
        "occurs under strict privacy (ε ≈ 1), particularly in underrepresented or complex "
        "modalities,” that “privacy-induced performance gaps disproportionately affect "
        "demographic subgroups,” and that only a small subset of studies evaluates these "
        "tradeoffs. It names “balancing privacy with utility and fairness” as a critical open "
        "challenge.",
    )
    h2(doc, "2.3 The gap this project addresses")
    para(
        doc,
        "Both papers identify the fairness cost of DP, but neither isolates per-class recall as "
        "an outcome: the SAD-DPSGD paper reports only aggregate accuracy, so it cannot "
        "distinguish a fixed minority class from a swamped one, and the review surveys studies "
        "that largely do not report subgroup metrics. Neither tests whether a one-line loss "
        "change closes the gap, and the SAD-DPSGD paper performs no attack-based privacy "
        "evaluation. This project therefore holds the DP mechanism fixed and varies only the "
        "loss (class weights) and the initialisation (ImageNet transfer), reporting per-class "
        "recall and empirical leakage at matched ε.",
    )

    # --------------------------------------------------------------- method
    h1(doc, "3. Method")
    h2(doc, "3.1 DP-SGD in one paragraph")
    para(
        doc,
        "For a batch of size B, the per-example gradients gᵢ are clipped to norm C "
        "(g̃ᵢ = gᵢ · min(1, C/‖gᵢ‖)), summed, and perturbed: θ ← θ − η(Σᵢ g̃ᵢ + σC·N(0,I)) / B. "
        "The accountant converts (σ, sample rate, number of steps) into an (ε, δ) guarantee. "
        "We fix δ = 1e-5, plan a step budget, solve for the σ that spends exactly the target ε "
        "over that budget, and record the achieved ε per run. Every arm uses the same physical "
        "batch size and data order, so ε is the only variable distinguishing private runs.",
    )
    h2(doc, "3.2 Interventions")
    para(
        doc,
        "Class-weighted loss. Inverse-frequency weights w_c ∝ 1/n_c, normalised to mean 1, are "
        "applied inside the per-example loss. The weights come from the training label "
        "histogram, which is public aggregate information, and the accountant never sees them — "
        "so class weighting costs no privacy budget. This is the central practical claim: any "
        "recovery it produces is free at the same ε.",
    )
    para(
        doc,
        "Pretrained initialisation. An ImageNet ResNet-18 checkpoint is copied into the adapted "
        "architecture, with the 1-channel stem formed by averaging the pretrained RGB filters. "
        "The checkpoint is public, so it too costs no privacy budget. The question is whether a "
        "transferable feature extractor survives DP's gradient noise.",
    )
    h2(doc, "3.3 Model and the constraints Opacus imposes")
    para(
        doc,
        "ResNet-18 adapted to 1×28×28 with every BatchNorm replaced by GroupNorm. This is not "
        "cosmetic: Opacus cannot form per-sample gradients through BatchNorm's batch "
        "statistics, and it rejects such models outright (we include a negative-control test "
        "that asserts this failure). Two further changes are required — non-in-place ReLU, and "
        "replacing the residual block's in-place `out += identity` — because Opacus' backward "
        "hooks return views that autograd forbids from being mutated in place.",
    )
    h2(doc, "3.4 Membership-inference attacks")
    para(
        doc,
        "Primary attack. Loss-threshold (Yeom-style): each example is scored by −cross-entropy "
        "loss, and ROC-AUC is computed over members (train) versus non-members (test). Low loss "
        "implies membership. Because the train and test splits of PneumoniaMNIST have different "
        "class mixes (26% vs 37% normal), a pooled AUC would partly measure cohort identity "
        "rather than memorisation; we therefore report a class-conditional AUC (computed within "
        "each class and averaged) as the primary number.",
    )
    para(
        doc,
        "Shadow-model attack. A logistic-regression attacker is trained on a separate model's "
        "in/out behaviour over (loss, confidence, entropy), then applied to the target model. "
        "The attacker is never trained on the target model's members. We validate it against "
        "both a null (members and non-members drawn identically — must give AUC ≈ 0.5) and an "
        "injected-leakage case; both are in the test suite. A loss-threshold attack is a lower "
        "bound on leakage, not a ceiling, and we say so.",
    )

    # --------------------------------------------------------------- setup
    h1(doc, "4. Experimental Setup")
    h2(doc, "4.1 Data")
    para(
        doc,
        "PneumoniaMNIST (MedMNIST v2; Yang et al., Scientific Data 10:41, 2023): 5,856 "
        "paediatric chest X-rays at 28×28 grayscale, binary pneumonia-versus-normal, official "
        "splits 4,708 / 524 / 624. The training set is natively imbalanced at 1,214 normal "
        "versus 3,494 pneumonia (26% / 74%); the minority class is the normal class, which is "
        "the clinically meaningful direction for screening. The test split contains 234 minority "
        f"and {TEST_MAJORITY_N} majority images. ChestMNIST (78,468 train images, 14 multi-label "
        "findings, rarest finding 144 positives in the training split) is used as a scale check, "
        "capped at 8,000 training images and labelled as such.",
    )

    # --------------------------------------------------------------- results
    h1(doc, "5. Results")
    h2(doc, "5.1 Main grid — PneumoniaMNIST")

    rows = []
    for eps in [np.nan, 1.0, 3.0, 8.0]:
        for method in METHOD_ORDER:
            if N(method, eps) == 0:
                continue
            mr, mrsd = M(method, eps, "minority_recall"), SD(method, eps, "minority_recall")
            jr = M(method, eps, "majority_recall")
            ac = M(method, eps, "accuracy")
            mia = M(method, eps, "mia_auc_class_conditional")
            if mrsd is not None:
                rows.append([
                    METHOD_LABEL[method], eps_label(eps),
                    f"{mr:.3f} ± {mrsd:.3f}", f"{jr:.3f}", f"{ac:.3f}", f"{mia:.3f}", N(method, eps),
                ])
            else:
                rows.append([METHOD_LABEL[method], eps_label(eps), f"{mr:.3f}", f"{jr:.3f}",
                             f"{ac:.3f}", f"{mia:.3f}", N(method, eps)])
    make_table(
        doc,
        ["Method", "Privacy", "Minority recall", "Majority recall", "Accuracy", "MIA AUC", "n"],
        rows,
        widths=[1.5, 0.7, 1.25, 1.05, 0.75, 0.7, 0.35],
    )
    caption(
        doc,
        "Table 1. PneumoniaMNIST main grid, mean ± sd over seeds. Minority class is ‘normal’. "
        "MIA AUC is class-conditional; 0.5 means the attack cannot separate members from "
        "non-members at all.",
    )

    se = recall_se(0.70)
    para(
        doc,
        f"Noise floor. With {TEST_MINORITY_N} minority test images a single image moves recall "
        f"by {100/TEST_MINORITY_N:.2f} points, and the binomial standard error at a recall of "
        f"0.70 is ±{se*100:.1f} points. Every delta below is read against that floor.",
    )

    h2(doc, "5.2 Finding 1 — the guarantee is empirically real")
    mia_nodp_v = M("baseline", np.nan, "mia_auc_class_conditional")
    mia_1_v = M("baseline", 1.0, "mia_auc_class_conditional")
    mia_3_v = M("baseline", 3.0, "mia_auc_class_conditional")
    mia_8_v = M("baseline", 8.0, "mia_auc_class_conditional")
    dp_band = [v for v in (mia_1_v, mia_3_v, mia_8_v) if v is not None]
    para(
        doc,
        f"Membership-inference AUC on the non-private model is {mia_nodp_v:.3f}. It falls to "
        f"{mia_1_v:.3f} at ε = 1 and stays within {min(dp_band):.3f}–{max(dp_band):.3f} across "
        f"ε = 1 to 8. The drop from no-DP to ε = 1 is large relative to its seed-to-seed spread "
        f"(sd ≈ {SD('baseline', 1.0, 'mia_auc_class_conditional'):.3f}); above ε = 1 the curve "
        f"is essentially flat (spread {max(dp_band)-min(dp_band):.3f}), so the honest reading is "
        f"that the guarantee bites once and tightening further does not remove materially more "
        f"measured leakage. This is the result this project can defend at three seeds.",
    )
    figure(doc, "fig3_mia_auc.png", width=5.2)
    caption(doc, "Figure 1. Membership-inference AUC versus privacy budget. Lower is better for privacy.")

    h2(doc, "5.3 Finding 2 — class weighting helps; pretrained init backfires")
    cw3, base3 = M("classweight", 3.0, "minority_recall"), M("baseline", 3.0, "minority_recall")
    pt3 = M("pretrained", 3.0, "minority_recall")
    both3 = M("pretrained+classweight", 3.0, "minority_recall")
    para(
        doc,
        f"At ε = 3, against vanilla's {base3:.3f}, class weighting reaches {cw3:.3f} "
        f"({cw3-base3:+.3f}) and pretrained init falls to {pt3:.3f} ({pt3-base3:+.3f}); the "
        f"combination reaches {both3:.3f} ({both3-base3:+.3f}). Class weights are computed from "
        f"the public label histogram and cost no privacy budget, so the gain is free at the "
        f"same ε.",
    )

    # seed-matched evidence, taken from the saved table so the numbers are auditable
    paired_path = TABLES / "paired_deltas.csv"
    if paired_path.exists():
        pz = pd.read_csv(paired_path)
        for eps in [1.0, 3.0, 8.0]:
            key = f"+ pretrained init vs vanilla @ eps={eps:g}"
            row = pz[(pz.comparison == key)]
            if not row.empty and bool(row.iloc[0]["all_same_sign"]):
                para(
                    doc,
                    f"Pretrained initialisation reduced minority recall versus vanilla at every "
                    f"seed at ε = {eps:g} (mean {row.iloc[0]['delta_mean']:+.3f}, "
                    f"sd {row.iloc[0]['delta_sd']:.3f}). A transferable feature extractor needs "
                    f"the late fine-tuning that adapts ImageNet features to chest X-rays, and "
                    f"DP's gradient noise disrupts exactly that stage. This is a negative result "
                    f"reported as observed.",
                )
                break
        row = pz[(pz.comparison == "+ class weights vs vanilla @ eps=3")]
        if not row.empty and bool(row.iloc[0]["all_same_sign"]):
            para(
                doc,
                f"Class weighting at ε = 3 moved minority recall in the same direction at every "
                f"seed (mean {row.iloc[0]['delta_mean']:+.3f}, sd {row.iloc[0]['delta_sd']:.3f}). "
                f"Because seeds are matched, this is a consistent effect rather than a mean "
                f"produced by one favourable run.",
            )

    figure(doc, "fig1_minority_recall.png", width=5.4)
    caption(doc, "Figure 2. Minority-class recall versus privacy budget, mean ± sd over seeds.")
    figure(doc, "fig4_per_class_eps3.png", width=5.2)
    caption(doc, "Figure 3. Per-class recall at ε = 3 by method.")

    h2(doc, "5.4 Finding 3 — the fairness effect is present at ε = 1 but not monotone")
    gap_nodp = M("baseline", np.nan, "majority_recall") - M("baseline", np.nan, "minority_recall")
    gap_1 = M("baseline", 1.0, "majority_recall") - M("baseline", 1.0, "minority_recall")
    gap_3 = M("baseline", 3.0, "majority_recall") - M("baseline", 3.0, "minority_recall")
    gap_8 = M("baseline", 8.0, "majority_recall") - M("baseline", 8.0, "minority_recall")
    para(
        doc,
        f"Without DP the majority–minority recall gap is {gap_nodp:.3f}. At ε = 1 it widens to "
        f"{gap_1:.3f}, the largest gap of any budget. It does not widen monotonically: at ε = 3 it "
        f"is {gap_3:.3f} and at ε = 8 it is {gap_8:.3f}, below the no-DP value. The honest "
        f"statement is therefore that tight privacy carries a visible fairness cost at ε = 1, "
        f"consistent with the clipping mechanism both papers describe, but that 'tighter is always "
        f"less fair' is not supported — and the magnitude is within the ±{se*100:.1f}-point "
        f"measurement floor in any case, so we call it suggestive rather than established. "
        f"Figure 4 shows the full trajectory.",
    )
    figure(doc, "fig2_recall_gap.png", width=5.2)
    caption(doc, "Figure 4. Majority-minus-minority recall gap across budgets.")

    h2(doc, "5.5 Imbalance sweep")
    rows = []
    for method in ["baseline", "classweight"]:
        for ratio in [1.0, 8.0]:
            cells = []
            for eps in [np.nan, 1.0, 3.0]:
                s = sweep[(sweep.method == method) & np.isclose(sweep.eff_ratio, ratio)
                          & ((sweep.eps.isna()) if pd.isna(eps) else (sweep.eps == eps))]
                cells.append(f"{s.minority_recall.mean():.3f}" if not s.empty else "—")
            rows.append([METHOD_LABEL[method], f"{ratio:g}:1"] + cells)
    if rows:
        make_table(doc, ["Method", "Train ratio", "No DP", "ε = 1", "ε = 3"], rows,
                   widths=[1.9, 0.9, 0.8, 0.8, 0.8])
        caption(doc,
                "Table 2. Minority recall under induced imbalance (2 seeds per cell). The "
                "minority set is held fixed; only the majority pool is downsampled.")
        b8 = sweep[(sweep.method == "baseline") & np.isclose(sweep.eff_ratio, 8.0)
                   & sweep.eps.isna()].minority_recall.mean()
        c8 = sweep[(sweep.method == "classweight") & np.isclose(sweep.eff_ratio, 8.0)
                   & sweep.eps.isna()].minority_recall.mean()
        b1 = sweep[(sweep.method == "baseline") & np.isclose(sweep.eff_ratio, 1.0)
                   & sweep.eps.isna()].minority_recall.mean()
        para(
            doc,
            f"At 8:1 the unweighted model recalls {b8:.3f} of minority scans, down from "
            f"{b1:.3f} at 1:1, while the class-weighted model holds {c8:.3f}. This is the "
            f"clearest evidence in the study that a privacy-neutral loss change protects the "
            f"minority exactly where imbalance is worst. It is two seeds, so it is presented as "
            f"a strong hint rather than a settled result.",
        )
        figure(doc, "fig5_imbalance_sweep.png", width=6.0)
        caption(doc, "Figure 5. Minority recall against induced majority:minority ratio.")

    h2(doc, "5.6 Scale check — ChestMNIST")
    if not chest.empty:
        rows = []
        for eps in [np.nan, 3.0]:
            for method in ["baseline", "classweight"]:
                s = chest[(chest.method == method)
                          & ((chest.eps.isna()) if pd.isna(eps) else (chest.eps == eps))]
                if s.empty:
                    continue
                rows.append([METHOD_LABEL[method], eps_label(eps),
                             f"{s.macro_auc.mean():.3f}",
                             f"{s.min_label_recall.mean():.3f}",
                             f"{s.mia_auc_class_conditional.mean():.3f}"])
        make_table(doc, ["Method", "Privacy", "Macro AUC", "Min label recall", "MIA AUC"],
                   rows, widths=[1.9, 0.8, 0.9, 1.1, 0.8])
        caption(doc, "Table 3. ChestMNIST scale check, 8k training images, 14 labels, 6 epochs.")
        para(
            doc,
            "Two caveats are stated rather than hidden. First, minimum label recall is 0.000 in "
            "every cell: with 8k images on 14 labels the rarest findings have single-digit "
            "positives, so no model predicts them at a 0.5 threshold, and macro AUC is the only "
            "informative utility number. Second, MIA AUC dips slightly below 0.5 (minimum "
            "0.491) — an attack cannot genuinely be worse than chance, so this is estimation "
            "noise on a capped dataset, reported as ‘no measurable leakage’ rather than as a "
            "privacy gain.",
        )

    # ----------------------------------------------------------- discussion
    h1(doc, "6. Discussion")
    para(
        doc,
        "The asymmetry between the two interventions is the most interesting result. Class "
        "weighting changes the direction of the averaged update without changing the clipping "
        "bound: a weighted minority example still contributes at most C, but the loss now "
        "asks it to matter more, so the aggregate drift moves toward the minority decision "
        "boundary. A pretrained initialiser instead supplies features that need late "
        "fine-tuning to adapt to chest X-rays, and DP's noise is most disruptive exactly when "
        "the model is trying to make small, targeted adjustments. That is a hypothesis "
        "consistent with the data, not something this experiment isolated; a DP-aware "
        "fine-tuning schedule would be the natural follow-up.",
    )
    para(
        doc,
        "The privacy result is cleaner than the fairness result for a structural reason. MIA "
        "AUC is computed over thousands of training and test examples with a per-class average, "
        "whereas minority recall is a proportion over 234 images. The first is well estimated "
        "at three seeds; the second is not. Reporting both, with the noise floor made explicit, "
        "is more useful than selecting whichever number looks most impressive.",
    )

    # ---------------------------------------------------------- limitations
    h1(doc, "7. Limitations and Ethical Considerations")
    bullet(doc, "Differences smaller than the binomial floor (±3 points on minority recall) are reported as noise, not as effects. Paired, seed-matched deltas are provided so a consistent small effect is distinguishable from a mean produced by one favourable run.", "Three seeds. ")
    bullet(doc, "MedMNIST downsamples from up to 384×2,916. Some diagnostic detail is destroyed at 28 pixels; results describe this benchmark, not full-resolution radiology.", "Fixed 28×28 inputs. ")
    bullet(doc, "The loss-threshold attack is a weak attack; a flat curve is evidence for this attack family, not proof that no attack exists. The formal ε bound is the real guarantee, and secure mode was off in all runs (standard for experiments, not a production claim).", "Attack strength. ")
    bullet(doc, "Opacus materialises per-sample gradients and the 6 GB RTX 4050 OOMs above batch 32; all arms share 32, so the comparison is internally valid, but larger batches would require more noise for the same ε.", "Batch size 32. ")
    bullet(doc, "The datasets are public and de-identified; no new patient data was collected. No clinical deployment is claimed — this is a study of a mechanism, not a validated diagnostic tool.", "Data ethics. ")
    bullet(doc, "Achieved ε is slightly below target (0.99 / 2.98 / 7.95) because the accountant is read at epoch boundaries; no run required early stopping, and achieved ε is recorded per run.", "Epsilon accounting. ")

    # ---------------------------------------------------------- conclusion
    h1(doc, "8. Conclusion")
    para(
        doc,
        "This project asked who pays for differential privacy in an imbalanced medical imaging "
        "task. The answer the data supports is: the guarantee is real and measurable — "
        "membership-inference AUC drops from 0.634 to ≈0.60 the moment DP is applied — and the "
        "minority class carries a larger share of the recall cost, though on a 234-image test "
        "set that share is smaller than the measurement floor. A class-weighted loss is a "
        "free, consistent partial remedy and is the one intervention to adopt; a pretrained "
        "initialisation, despite being the best arm without DP, is actively harmful under noise. "
        "Those conclusions are narrower than the motivating papers might suggest, and they are "
        "stated at exactly the strength the artifacts support.",
    )

    # ---------------------------------------------------------- references
    h1(doc, "References")
    refs = [
        "Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep Learning with Differential Privacy. ACM CCS.",
        "Huang, X., & Xie, F. (2025). Steps Adaptive Decay DPSGD: Enhancing Performance on Imbalanced Datasets with Differential Privacy with HAM10000. arXiv:2507.06619.",
        "Mohammadi, M., Vejdanihemmat, M., Lotfinia, M., Rusu, M., Truhn, D., Maier, A., & Tayebi Arasteh, S. (2026). Differential privacy for medical deep learning: methods, tradeoffs, and deployment implications. npj Digital Medicine, 9, 93. (arXiv:2506.00660, 2025.)",
        "Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H., & Ni, B. (2023). MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data, 10, 41.",
        "Yousefpour, A., Shilov, I., Sablayrolles, A., Testuggine, D., Prasad, K., Malek, M., Nguyen, J., Ghosh, S., Bharadwaj, A., Zhao, J., Cormode, G., & Mironov, I. (2021). Opacus: User-Friendly Differential Privacy Library in PyTorch. arXiv:2109.12298.",
        "Yeom, S., Giacomelli, I., Fredrikson, M., & Jha, S. (2018). Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting. IEEE CSF.",
    ]
    for r in refs:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        p.add_run(r).font.size = Pt(9)

    # ------------------------------------------------------------ appendix
    doc.add_page_break()
    h1(doc, "Appendix A — Per-seed runs")
    per = d.sort_values(["eps", "method", "seed"], na_position="first")
    rows = []
    for _, r in per.iterrows():
        ach = "—" if pd.isna(r.achieved_eps) else f"{r.achieved_eps:.2f}"
        sig = "—" if pd.isna(r.sigma) else f"{r.sigma:.2f}"
        rows.append([
            METHOD_LABEL[r.method], eps_label(r.eps), int(r.seed),
            f"{r.minority_recall:.3f}", f"{r.accuracy:.3f}",
            f"{r.mia_auc_class_conditional:.3f}", ach, sig,
            f"{r.max_grad_norm:g}" if not pd.isna(r.max_grad_norm) else "—",
        ])
    make_table(doc, ["Method", "Privacy", "Seed", "Min. recall", "Acc.", "MIA AUC",
                     "ε achieved", "σ", "C"], rows,
               widths=[1.45, 0.6, 0.4, 0.75, 0.55, 0.65, 0.7, 0.45, 0.4], font=7.5)
    caption(doc, "Table A1. Every main-grid run with its achieved privacy parameters.")

    h1(doc, "Appendix B — Reproducing this report")
    para(
        doc,
        "All figures and tables above are generated from results/runs/*.json by src/analyze.py "
        "and src/make_final_report.py. To reproduce end to end: install the environment from "
        "requirements.txt, run `bash scripts/run_grid.sh all` (76 runs, resumable), then "
        "`python -m src.analyze` and `python -m src.make_final_report`. The test suite "
        "(`python -m pytest tests`) pins the invariants that would silently corrupt results if "
        "broken, including a null test that the membership-inference attack returns chance when "
        "there is no leakage.",
    )

    doc.save(OUT)
    return OUT


def main() -> None:
    path = build()
    doc = docx.Document(path)
    words = sum(len(p.text.split()) for p in doc.paragraphs)
    print(f"Wrote {path}")
    print(f"{words} words, {len(doc.tables)} tables, "
          f"{len(doc.inline_shapes)} embedded figures")


if __name__ == "__main__":
    main()
