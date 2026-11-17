# Submission checklist

Four deliverables plus a guidebook, all generated from the run artifacts:
`docs/PROPOSAL.docx` (one page), `docs/FINAL_REPORT.docx` (~2,850 words),
`docs/PRESENTATION.pptx` (15 slides, with PDF), and `docs/GUIDEBOOK.docx`
(14-page handbook) with a live companion at `docs/guidebook/index.html`.

## 0. What to fill in

Nothing. Every document carries the identity **Tushar Sharma · MTech I Year,
NSUT · tushar.sharma.pg26@nsut.ac.in**, set once in the generator sources and
rendered into the proposal, report, guidebook, deck and results summary. If the
details ever change, edit the strings in `src/make_*.py` and regenerate; do not
hand-edit the `.docx` files.

## 1. One-page proposal (due now)

- [x] `docs/PROPOSAL.docx` — **generated and fits on one page** (741 words; Word's
      own PDF export is 1 page, content ending ~645 pt down a 792 pt page). Rebuild with:
      ```bash
      .venv/Scripts/python.exe -m src.make_proposal
      ```
- [x] Identity filled in at the top: name, MTech year and department, NSUT email.
- [x] Citations verified against the arXiv full text, with every quoted phrase checked
      character-for-character:
      Huang & Xie arXiv:2507.06619 (9 Jul 2025 — the "works well on large datasets /
      fail on small, imbalanced" clause and the +2.15% at ε=3 claim both confirmed), and
      Mohammadi et al. arXiv:2506.00660 → *npj Digital Medicine* 9:93 (2026)
      (74 studies; "severe degradation ... under strict privacy (ε≈1)";
      "privacy-induced performance gaps disproportionately affect demographic subgroups";
      "balancing privacy with utility and fairness" — all verbatim).
- [x] Reference implementations explored, per the guidelines: Opacus
      (`meta-pytorch/opacus`), MedMNIST (`MedMNIST/MedMNIST`), ML Privacy Meter
      (`privacytrustlab/ml_privacy_meter`) and a PyTorch shadow-model MIA
      (`yonsei-sslab/MIA`). SAD-DPSGD has **no released code**, and no established repo
      applies DP-SGD to chest X-rays on MedMNIST — both facts are stated in the proposal.

## 2. Final report

- [x] `docs/FINAL_REPORT.docx` — full academic report built from the artifacts:
      abstract, introduction, related work and gap, method, setup, results with
      four tables and five embedded figures, discussion, limitations/ethics,
      conclusion, references, and two appendices (per-seed run table; repro steps).
      Rebuild with:
      ```bash
      .venv/Scripts/python.exe -m src.make_final_report
      ```
- [x] Identity line under the title is filled in.

## 3. Presentation (November 20, per the guidelines)

- [x] `docs/PRESENTATION.pptx` — 15 slides, 16:9, rebuilt from artifacts and
      **visually inspected slide by slide** (rendered through PowerPoint to
      `docs/render/`). Rebuild and re-render with:
      ```bash
      .venv/Scripts/python.exe -m src.make_deck
      .venv/Scripts/python.exe -m src.render_deck --pdf
      ```
- [x] `docs/PRESENTATION.pdf` — handout copy.

The deck's running order:

1. Title
2. DP-SGD protects patients by discarding signal — the problem
3. Two papers name the problem; neither tests a fix
4. PneumoniaMNIST: a natively imbalanced task
5. Four budgets × four methods × three seeds — setup
6. **Result 1** — the guarantee is empirically real (MIA 0.634 → 0.596)
7. **Result 2** — class weighting helps; pretrained init backfires
8. Per-class view at ε = 3
9. **Result 3** — fairness effect present at ε = 1 but not monotone
10. Imbalance sweep — weighting absorbs the damage
11. Scale check (ChestMNIST), with its caveats stated
12. Limitations, stated before you are asked
13. What this establishes
14. What I would do next
15. Thank you

### Numbers to have memorised

From `docs/FINAL_REPORT.docx` §5: MIA AUC 0.634 (no DP) → 0.596 (ε=1); minority
recall at ε=3 is 0.685 vanilla, 0.735 class-weighted, 0.652 pretrained; at 8:1
imbalance 0.622 → 0.729 with class weights. Always say "over three seeds", and for
uncertainty point at `results/tables/paired_deltas.csv`.

## 4. Guidebook (handbook + live version)

- [x] `docs/GUIDEBOOK.docx` and `GUIDEBOOK.pdf` — 14-page printable handbook
      covering everything: question, papers and gap, data, method, tools,
      results, limitations, reproduction, artifact schema, live-version options,
      references, deliverable map and viva prep. Rebuild with:
      ```bash
      .venv/Scripts/python.exe -m src.make_guidebook_docx
      ```
- [x] `docs/guidebook/index.html` — the live interactive edition. Same content
      plus a filterable/sortable table of all 76 runs. Self-contained (figures
      embedded), works offline. It was loaded in a real browser and its filters
      checked against the artifacts (48 main / 24 sweep / 4 ChestMNIST).
      Rebuild with:
      ```bash
      .venv/Scripts/python.exe -m src.make_guidebook
      ```
- [ ] Run it when you present: `scripts/serve_guidebook.py`, then open
      `http://127.0.0.1:8000`.
- [ ] Optional — publish it: push the repo, then repo **Settings → Pages →
      Deploy from a branch → main / docs**. The guidebook lands at
      `https://<your-github-username>.github.io/dp-sgd-chest-xray-fairness/guidebook/`;
      Settings → Pages displays the exact URL once it is live. Only do this if
      the author is happy for the repo (and their name) to be public.

## 5. GitHub repository (guidelines item 1 + 3)

This folder is already a git repository with one commit, made under the identity
**Tushar Sharma <tushar.sharma.pg26@nsut.ac.in>**. Create an empty repository on
GitHub named `dp-sgd-chest-xray-fairness` (no README, no .gitignore — this folder
already has both), then point it at the remote and push:

```bash
cd dp-sgd-chest-xray-fairness
git remote add origin https://github.com/<your-github-username>/dp-sgd-chest-xray-fairness.git
git push -u origin main
```

`data/` is ignored (via `.gitignore`), so the 83 MB of `.npz` images are not
uploaded; the `medmnist` package re-downloads them on demand. The
`results/runs/*.json` artifacts **are** committed — they are small and they are
the evidence for every number in the report.


## 6. Viva questions to expect

- *Why not report accuracy?* Because predicting pneumonia for everything scores
  74% while getting every normal scan wrong. That is in the test suite
  (`test_majority_class_collapse_scores_high_accuracy_but_zero_minority_recall`).
- *Is the MIA convincing?* No — say so. It is a lower bound; the formal ε is the
  guarantee. The attack shows the guarantee bites; it does not prove no attack exists.
- *Why GroupNorm?* Opacus cannot form per-sample gradients through BatchNorm's batch
  statistics. `test_privacy_engine_rejects_batchnorm` demonstrates the failure.
- *Is class weighting cheating on privacy?* No: weights come from the training label
  histogram, which is public aggregate information, and the accountant never sees
  them. Same ε, same guarantee.
- *Why does ε=1 run fewer effective steps?* It does not stop early here; achieved ε is
  recorded per run. At ε=1 the noise multiplier is 1.40 vs 0.60 at ε=8, which is the
  actual cost mechanism.
- *Why does pretrained init hurt under DP?* It is the best arm without DP and the worst at
  ε=8, at every seed. Transfer needs the late fine-tuning that adapts ImageNet features to
  chest X-rays, and that is exactly what gradient noise disrupts. Say this is a hypothesis
  consistent with the data, not something the experiment isolated — a DP-aware schedule
  would be the follow-up.
- *Only 3 seeds?* Yes. That is why `paired_deltas.csv` exists: seeds are matched, so a
  delta that holds at every seed is stronger than two overlapping means. Any difference
  smaller than ±3 points on minority recall is called noise, not a finding.

## 7. Final checks before you submit

```bash
.venv/Scripts/python.exe -m pytest tests -q              # 22 invariant tests
.venv/Scripts/python.exe -m src.analyze                  # tables + figures + RESULTS.md
.venv/Scripts/python.exe -m src.report                   # docs/REPORT.md
.venv/Scripts/python.exe -m src.make_proposal            # docs/PROPOSAL.docx
.venv/Scripts/python.exe -m src.make_final_report        # docs/FINAL_REPORT.docx
.venv/Scripts/python.exe -m src.make_deck                # docs/PRESENTATION.pptx
.venv/Scripts/python.exe -m src.render_deck --pdf        # PNGs + PRESENTATION.pdf
.venv/Scripts/python.exe -m src.make_guidebook           # docs/guidebook/index.html
.venv/Scripts/python.exe -m src.make_guidebook_docx      # docs/GUIDEBOOK.docx
.venv/Scripts/python.exe -m src.runner --list            # every run on disk
```

The one thing not to do: hand-edit a number in any generated document.
Regenerate it. If a run is missing the generator prints `MISSING`, and that is
better than a number nobody can reproduce.
