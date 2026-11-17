# Who Pays for Differential Privacy?

**Per-class fairness and membership leakage in differentially private chest X-ray classification**

**Tushar Sharma** · MTech I Year · NSUT · tushar.sharma.pg26@nsut.ac.in
Course: Machine Learning Project — Ethics / Responsible AI track · November 2026

---

Differential privacy is the standard guarantee for training on sensitive medical data, and DP-SGD is
its most-used mechanism. This project asks **who pays the cost of that guarantee**. We train the same
model at privacy budgets ε = {1, 3, 8} (δ = 1e-5) and without DP on PneumoniaMNIST (with a ChestMNIST
scale check), and ask three questions:

1. Does DP-SGD reduce recall for the **minority class** more than for the majority class?
2. Do cheap, privacy-neutral fixes — a **class-weighted loss** or **pretrained initialisation** — recover it?
3. Does the privacy guarantee actually **reduce measurable membership leakage** (empirical, not just theoretical)?

Every number in `results/RESULTS.md` is generated from the run artifacts in `results/runs/`.

---

## Headline results (76 runs, RTX 4050 Laptop GPU, ~3 hours total)

| Finding | Number |
|---|---|
| The guarantee is empirically real | membership-inference AUC **0.634 → 0.596** at ε = 1, with seed spread an order of magnitude smaller than the effect |
| Class weighting is a free, consistent partial remedy | minority recall **+0.050** at ε = 3 (all three seeds), and **0.622 → 0.729** at 8:1 induced imbalance |
| Pretrained init is *not* a remedy under DP | best arm without DP (0.739), worse than vanilla at every seed at ε = 8 — reported as a negative result |
| The fairness effect is present but not monotone | majority–minority recall gap 0.262 (no DP) → 0.287 at ε = 1 → 0.236 at ε = 8; within the ±3-point floor below |

The test split holds only 234 minority images, so a single image moves minority recall by 0.43 points.
Differences smaller than the ±3-point binomial floor are reported as **noise, not findings**. The
membership-inference result is the finding this project defends at three seeds; the recall deltas are
reported with their uncertainty.

---

## Documents

| Deliverable | File |
|---|---|
| One-page proposal (submission) | `docs/PROPOSAL.docx` / `.pdf`, source `docs/PROPOSAL.md` |
| Full project report (12 pages) | `docs/FINAL_REPORT.docx` / `.pdf` |
| Guidebook — printable (14 pages) | `docs/GUIDEBOOK.docx` / `.pdf` |
| Guidebook — live/interactive | `docs/guidebook/index.html` (open in a browser, works offline) |
| Presentation (15 slides) | `docs/PRESENTATION.pptx` / `.pdf` |
| Results summary | `docs/REPORT.md`, `results/RESULTS.md` |
| Submission checklist and viva prep | `docs/SUBMISSION_CHECKLIST.md` |

All documents are **generated from the run artifacts**, never hand-edited. To rebuild:

```bash
.venv/Scripts/python.exe -m src.make_proposal       # one-page proposal
.venv/Scripts/python.exe -m src.make_final_report   # full report
.venv/Scripts/python.exe -m src.make_guidebook_docx # printable guidebook
.venv/Scripts/python.exe -m src.make_guidebook      # live guidebook (HTML)
.venv/Scripts/python.exe -m src.make_deck           # presentation
.venv/Scripts/python.exe -m src.report              # results summary
```

---

## Research context

| Source | Claim used here |
|---|---|
| Huang & Xie, *Steps Adaptive Decay DPSGD*, [arXiv:2507.06619](https://arxiv.org/abs/2507.06619) (Jul 2025) | Gradient-noise DP "works well on large datasets like MNIST and CIFAR-100, but fail[s] on small, imbalanced medical datasets like HAM10000"; minority gradients get clipped while majority classes dominate. |
| Mohammadi et al., *Differential privacy for medical deep learning*, [arXiv:2506.00660](https://arxiv.org/abs/2506.00660) (May 2025); *npj Digital Medicine* 9:93 (2026) | Scoping review of 74 studies: "severe degradation often occurs under strict privacy (ε≈1), particularly in underrepresented or complex modalities"; "privacy-induced performance gaps disproportionately affect demographic subgroups"; "balancing privacy with utility and fairness" is a critical challenge. |
| Yang et al., *MedMNIST v2*, [Scientific Data 10:41](https://doi.org/10.1038/s41597-022-01721-8) (2023) | Dataset construction; ResNet-18 (28px) reference on PneumoniaMNIST: **85.4% ACC / 0.944 AUC**. |

**Gap addressed:** both papers identify the fairness cost of DP but neither isolates *per-class recall*
as an outcome, and neither tests whether a one-line loss change closes the gap. We do both, and we add
an attack-based privacy evaluation that the SAD-DPSGD paper lacks.

### Reference implementations explored

| Repository | What it provides |
|---|---|
| [`meta-pytorch/opacus`](https://github.com/meta-pytorch/opacus) | DP-SGD with per-sample / ghost clipping and the RDP accountant — the mechanism used here |
| [`MedMNIST/MedMNIST`](https://github.com/MedMNIST/MedMNIST) | Official dataset loaders and splits |
| [`privacytrustlab/ml_privacy_meter`](https://github.com/privacytrustlab/ml_privacy_meter) | Privacy-auditing library for membership inference |
| [`yonsei-sslab/MIA`](https://github.com/yonsei-sslab/MIA) | PyTorch shadow-model implementation of Shokri et al. |

No official code exists for the SAD-DPSGD paper, and no established repository applies DP-SGD to chest
X-rays on MedMNIST — only small unofficial attempts exist. The pipeline here is therefore built directly
on Opacus primitives rather than a fork, which keeps the DP mechanics auditable.

---

## Datasets

**PneumoniaMNIST** — 5,856 paediatric chest X-rays, 28×28 grayscale, binary. Official split used unchanged:

| Split | n | normal (class 0) | pneumonia (class 1) |
|---|---|---|---|
| train | 4,708 | **1,214 (26%)** | 3,494 (74%) |
| val | 524 | 135 | 389 |
| test | 624 | 234 | 390 |

The **minority class is "normal"**. This is the clinically meaningful direction: in screening, a normal
scan misread as pneumonia triggers unnecessary follow-up, so minority recall is the harm-relevant metric.

Note that test is *less* imbalanced than train (37% vs 26% normal). A naive pooled membership-inference
AUC would partly measure this distribution shift, so we report a **class-conditional** AUC as the primary
privacy number and keep the pooled AUC only as a diagnostic.

**ChestMNIST** — 78,468 train images, 14 multi-label findings, rarest finding has 144 positives in the
training split (0.18%). Used as a scale/severe-imbalance check, capped at 8,000 training images for
tractability (labelled as such in the artifact).

Both datasets are public, de-identified, and downloaded automatically by the `medmnist` package. They are
**not committed** to this repository (`data/` is gitignored) and require no data-use agreement.

---

## Methods compared

All arms share architecture, optimiser, data order, batch size and epoch count. Only the mechanism and the
two interventions vary:

| Arm | DP | Class weights | Pretrained init |
|---|---|---|---|
| `baseline` | ✓ when ε set | ✗ | ✗ |
| `classweight` | ✓ when ε set | ✓ inverse-frequency | ✗ |
| `pretrained` | ✓ when ε set | ✗ | ✓ ImageNet ResNet-18 |
| `pretrained+classweight` | ✓ when ε set | ✓ | ✓ |

**Why these two fixes are privacy-neutral.** Class weights are computed from the training label histogram
(public aggregate). ImageNet weights are public. Neither consumes privacy budget, so any recovery they
produce is genuinely free at the same ε — that is the practical claim of the project.

**Architecture.** ResNet-18 adapted to 1×28×28, with every BatchNorm replaced by GroupNorm. This is not
cosmetic: Opacus cannot form per-sample gradients through BatchNorm's batch statistics, and the residual
block's in-place `out += identity` is rejected by autograd once Opacus' hooks return a view. Both are
handled explicitly in `src/models.py` and documented there.

---

## How DP-SGD is run (and what could invalidate it)

- **Accounting.** Opacus RDP accountant. We plan a fixed step budget, solve for the noise multiplier σ that
  spends exactly the target ε over that budget, then **stop early if the accountant crosses ε**. Each
  artifact records the *achieved* ε, σ, clipping norm C, and steps actually taken.
- **Clipping bound C** is calibrated from the **median true per-sample gradient norm** measured once on an
  untrained model with Opacus' own `grad_sample` buffers. The median (not max) keeps noise low. This
  statistic is aggregated, computed before training, and releases nothing per example.
- **Batch size 32 is mandatory for DP arms** on the 6 GB RTX 4050: Opacus materialises per-sample
  gradients, and batch 64 fragmented reserved memory to OOM around step 60 on this Windows build
  (`expandable_segments` is unsupported here). Non-private arms use the same 32 so the comparison is clean.
- **Secure mode is off** in all runs (`secure_mode=False`). This is standard for experimentation, but it
  means the noise is drawn from a fast PRNG. A production claim would require `secure_mode=True`; we make
  no such claim.
- **δ = 1e-5**, fixed for every run and recorded in the config.
- **Off-by-one honesty.** Epsilon is *reported as achieved*, not as requested. Where a run stops early, the
  achieved value can be slightly below target because the accountant is evaluated only at epoch boundaries.

---

## Membership-inference attack

Primary attack: **loss-threshold (Yeom-style)**. Score each example by `-cross-entropy loss`; low loss ⇒
more likely a training member. Reported as ROC-AUC over members (train) vs non-members (test).

- `mia_auc_class_conditional` — AUC computed *within each class* and averaged. **This is the primary number.**
  It is immune to the train/test class-mix difference.
- `mia_auc_pooled` — diagnostic only; contaminated by the split difference noted above.
- `mia_tpr_at_1pct_fpr`, `mia_tpr_at_10pct_fpr` — attack power at low false-positive rates.

An AUC near 0.5 means the attack cannot distinguish members from non-members. **Lower is better for
privacy.** A `learned_attack` (logistic regression on loss/confidence/entropy) is also implemented for a
shadow-model variant.

Caveat we state rather than hide: a loss-threshold attack is **weak**. A *flat* AUC under DP is evidence of
reduced leakage for *this attack family*, not proof that no attack exists. The formal ε guarantee is the
real bound; the attack is a sanity check that it bites in practice.

---

## Quick start

```bash
# from the repository root, Windows + Git Bash
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python.exe -m pip install -r requirements.txt

# resilient full run: 48 main + 24 imbalance + 4 ChestMNIST = 76 runs.
# Each run is its own short-lived process and is skipped if already on disk.
bash scripts/run_grid.sh all

# aggregate into tables, figures and results/RESULTS.md
.venv/Scripts/python.exe -m src.analyze

# build the documents from the artifacts
.venv/Scripts/python.exe -m src.report
.venv/Scripts/python.exe -m src.make_proposal
```

Prefer `scripts/run_grid.sh` over `python -m src.runner --grid all`: the runner spawns a child per run but
is itself one long-lived process, so if that parent is killed the batch dies with it. The shell driver
inverts that nesting.

Runs are **idempotent and resumable**: each is keyed by a hash of its config, so re-running skips completed
work. `--force` reruns. `--list` prints everything on disk.

The `results/runs/*.json` artifacts are committed, so `src.analyze`, `src.report` and every document
generator work **without re-running any training**.

---

## Repository layout

```
src/
  data.py               dataset loading, imbalance construction, class weights
  models.py             GroupNorm ResNet-18, Opacus-safe forward, ImageNet transfer
  train.py              training engine, DP-SGD (ghost clipping), C calibration, early stop
  metrics.py            per-class recall, recall gap, macro/weighted aggregation
  mia.py                loss-threshold and shadow-model membership-inference attacks
  runner.py             experiment grid, JSON artifacts, resume, per-run process isolation
  analyze.py            tables, figures, RESULTS.md — all from artifacts
  report.py             docs/REPORT.md — every claim computed from artifacts
  make_proposal.py      builds the one-page docs/PROPOSAL.docx
  make_final_report.py  builds docs/FINAL_REPORT.docx
  make_guidebook_docx.py builds docs/GUIDEBOOK.docx (printable)
  make_guidebook.py     builds docs/guidebook/index.html (live)
  make_deck.py          builds docs/PRESENTATION.pptx
  render_deck.py        renders the deck to PNGs for visual inspection
scripts/
  run_grid.sh           resilient shell driver (one fresh process per run)
  serve_guidebook.py    serves the live guidebook on 127.0.0.1:8000
tests/
  test_invariants.py    22 tests: weights, MIA null/leak, Opacus constraints, loss shapes
docs/                   proposal, report, guidebook (print + live), deck, checklist
results/
  runs/                 one JSON per run (config, history, metrics, achieved eps)
  tables/               CSV aggregates incl. paired_deltas.csv
  figures/              6 PNG figures
  RESULTS.md            generated summary
data/                   empty — datasets download on demand (medmnist)
```

---

## Limitations (stated up front)

1. **Single GPU, modest grid.** 3 seeds on the main grid. Differences smaller than seed-to-seed spread are
   reported as noise, not findings.
2. **28×28 images.** MedMNIST downsamples from 384×2,916. Some clinically relevant detail is destroyed at
   28px; results speak to the benchmark, not to full-resolution radiology.
3. **One architecture, one optimiser (AdamW), one δ.** No claim of generality across model families.
4. **Loss-threshold MIA is a lower bound on leakage**, not a ceiling.
5. **ChestMNIST capped at 8k images** and 6 epochs; it is a transfer check, not a full-data result.
6. **Early stopping at target ε** means DP arms may see fewer samples than their epoch budget implies. Step
   counts are recorded so the comparison is inspectable.
7. **No clinical deployment claim.** This is an ethics/responsible-AI study of a mechanism, not a validated
   diagnostic tool.

---

## Citation

```bibtex
@misc{huang2025saddpsgd,
  title  = {Steps Adaptive Decay DPSGD: Enhancing Performance on Imbalanced Datasets with
            Differential Privacy with HAM10000},
  author = {Huang, Xiaobo and Xie, Fang},
  year   = {2025},
  eprint = {2507.06619},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

@article{mohammadi2026differential,
  title   = {Differential privacy for medical deep learning: methods, tradeoffs, and
             deployment implications},
  author  = {Mohammadi, Marziyhe and Vejdanihemmat, Mohsen and Lotfinia, Mahshad and
             Rusu, Mirabela and Truhn, Daniel and Maier, Andreas and Tayebi Arasteh, Soroosh},
  journal = {npj Digital Medicine},
  volume  = {9},
  pages   = {93},
  year    = {2026},
  doi     = {10.1038/s41746-025-02280-z}
}

@article{yang2023medmnist,
  title   = {MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical
             image classification},
  author  = {Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and
             Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
  journal = {Scientific Data},
  volume  = {10},
  pages   = {41},
  year    = {2023},
  doi     = {10.1038/s41597-022-01721-8}
}
```

## License

Code released under the MIT License (see `LICENSE`). Datasets retain their original licenses
(MedMNIST: CC BY 4.0; derived from Kermany et al. and NIH-ChestXray14) — cite them, and do not
redistribute clinical images.
