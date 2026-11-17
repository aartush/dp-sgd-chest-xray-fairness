"""Generate the one-page proposal as a .docx for submission.

Kept as a script rather than a one-off command so the document can be rebuilt
after any edit to docs/PROPOSAL.md's content. The layout is tuned to fit one
page at 9.5 pt with 0.5-0.6 inch margins, per the course guideline.

Usage:  python -m src.make_proposal
Writes: docs/PROPOSAL.docx
"""

from __future__ import annotations

from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "PROPOSAL.docx"

TITLE = "Machine Learning Project Proposal — Ethics / Responsible AI track"
HEADER = (
    "Tushar Sharma  ·  MTech I Year, NSUT  ·  tushar.sharma.pg26@nsut.ac.in   ·   Date: 17 Nov 2026"
)

SECTIONS = [
    (
        "1. Problem statement",
        [
            (
                None,
                "Differential privacy (DP) is the standard guarantee for training on sensitive "
                "medical data, and DP-SGD is its most-used mechanism. The question this project "
                "asks is who pays for that guarantee. DP-SGD clips each per-example gradient to a "
                "fixed norm before averaging and adding noise. Clipping is what bounds one "
                "patient's influence on the model, but it also erases the gradient signal of rare "
                "examples, whose unclipped updates are the ones most likely to exceed the bound. "
                "In an imbalanced clinical dataset the rare group is usually a clinically "
                "important minority (here, the normal class). The research question is therefore: "
                "does DP-SGD degrade recall for minority classes disproportionately, and can "
                "cheap, privacy-neutral interventions — a class-weighted loss or a pretrained "
                "initialisation — recover that loss without weakening the guarantee? Per-class "
                "recall and empirical leakage (a membership-inference attack) are measured at "
                "ε = {1, 3, 8} and without DP, so the utility–fairness–privacy trade-off is "
                "readable from one table."
            )
        ],
    ),
    (
        "2. Selected papers and research gaps",
        [
            (
                "Huang & Xie, “Steps Adaptive Decay DPSGD”, arXiv:2507.06619 (Jul 2025). ",
                "Gradient-noise DP methods “work well on large datasets like MNIST and CIFAR-100, "
                "but fail on small, imbalanced medical datasets like HAM10000,” because minority "
                "gradients are clipped while majority classes dominate the update. Their remedy "
                "changes the training schedule (linearly decaying noise and clipping) and reports "
                "+2.15% accuracy at ε = 3. Gap: only aggregate accuracy is reported, so it cannot "
                "show whether the minority class was fixed or swamped; no loss-level or "
                "initialisation intervention is tested; it is a two-author preprint with no peer "
                "review and no attack-based privacy evaluation.",
            ),
            (
                "Mohammadi et al., “Differential privacy for medical deep learning”, "
                "arXiv:2506.00660 (May 2025); npj Digital Medicine 9:93 (2026). ",
                "A scoping review of 74 studies concluding that “severe degradation often occurs "
                "under strict privacy (ε ≈ 1), particularly in underrepresented or complex "
                "modalities,” that “privacy-induced performance gaps disproportionately affect "
                "demographic subgroups,” and that only a small subset of studies evaluates these "
                "tradeoffs at all. It names “balancing privacy with utility and fairness” as a "
                "critical open challenge. Gap: a review can identify the tension but cannot test "
                "a fix; no study isolates minority-class recall as the outcome, and none tests "
                "whether class weighting closes it.",
            ),
            (
                "This project's contribution: ",
                "Hold the DP mechanism fixed and vary only the loss (class weights) and the "
                "initialisation (ImageNet transfer), reporting per-class recall and a "
                "membership-inference attack rather than aggregate accuracy. Class weighting uses "
                "only public class frequencies and pretrained init uses public ImageNet weights, "
                "so neither consumes privacy budget — any recovery is free at the same ε.",
            ),
        ],
    ),
    (
        "3. Dataset",
        [
            (
                "PneumoniaMNIST (primary). ",
                "5,856 paediatric chest X-rays, 28×28 grayscale, binary pneumonia-vs-normal, from "
                "MedMNIST v2 (Yang et al., Scientific Data 10:41, 2023). Official splits "
                "4,708 / 524 / 624. Genuinely imbalanced: train has 1,214 normal vs 3,494 "
                "pneumonia (26% / 74%), not synthetically induced. ChestMNIST (78,468 frontal "
                "X-rays, 14 multi-label findings, rarest finding only 144 positives in the "
                "training split) is used as a scale check. Both are public, de-identified, and "
                "download automatically via the medmnist package; no data-use agreement is "
                "required.",
            )
        ],
    ),
    (
        "4. Software and tools",
        [
            (
                None,
                "Python 3.12, PyTorch 2.6 (CUDA 12.4) and torchvision for training and ImageNet "
                "initialisation; Opacus 1.5.4 for DP-SGD with the RDP accountant (GroupNorm "
                "replaces BatchNorm so per-sample gradients are valid); scikit-learn for "
                "per-class metrics and the attack classifier; matplotlib and pandas for analysis. "
                "Hardware: NVIDIA RTX 4050 Laptop GPU (6 GB), which forces batch size 32 for DP "
                "arms because Opacus materialises per-sample gradients — this constraint is "
                "reported with every run. Each run writes a JSON artifact recording seed, "
                "achieved ε, noise multiplier, clipping norm, steps and full history, so every "
                "reported number is traceable.",
            ),
            (
                "Existing implementations explored. ",
                "Meta's Opacus (github.com/meta-pytorch/opacus) supplies per-sample clipping, "
                "ghost clipping and the RDP accountant; MedMNIST (github.com/MedMNIST/MedMNIST) "
                "supplies the official loaders and splits; ML Privacy Meter "
                "(github.com/privacytrustlab/ml_privacy_meter) and a PyTorch shadow-model "
                "implementation (github.com/yonsei-sslab/MIA) informed the attack design. The "
                "SAD-DPSGD authors released no code, and no established repo applies DP-SGD to "
                "chest X-rays on MedMNIST, so the pipeline is built directly on Opacus.",
            ),
        ],
    ),
    (
        "5. Deliverables",
        [
            (
                None,
                "Trained models at ε = {1, 3, 8} and no-DP across four methods (vanilla, "
                "class-weighted, pretrained, both); per-class recall and recall-gap tables; "
                "membership-inference AUC showing the empirical privacy gain; an imbalance-ratio "
                "sweep; and a one-page results summary. All runs complete on the available GPU in "
                "a few hours.",
            )
        ],
    ),
]


def build() -> Path:
    doc = docx.Document()
    sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = Inches(0.5)
    sec.left_margin = sec.right_margin = Inches(0.6)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(9.5)
    normal.paragraph_format.space_after = Pt(2)
    normal.paragraph_format.line_spacing = 1.0

    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = t.add_run(TITLE)
    run.bold = True
    run.font.size = Pt(12)

    m = doc.add_paragraph()
    m.alignment = WD_ALIGN_PARAGRAPH.CENTER
    m.add_run(HEADER).font.size = Pt(8.5)

    for heading, paras in SECTIONS:
        hp = doc.add_paragraph()
        hr = hp.add_run(heading)
        hr.bold = True
        hr.font.size = Pt(10.5)
        hp.paragraph_format.space_before = Pt(3)

        for lead, body in paras:
            p = doc.add_paragraph()
            if lead:
                p.add_run(lead).bold = True
            p.add_run(body)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    return OUT


def main() -> None:
    path = build()
    d = docx.Document(path)
    words = sum(len(p.text.split()) for p in d.paragraphs)
    print(f"Wrote {path}")
    print(f"{words} words; estimated pages at 9.5pt/0.5in margins: {words/800:.2f}")


if __name__ == "__main__":
    main()
