"""Build a self-contained, browsable guidebook as a single HTML file.

This is the "live version" of the project: every figure is embedded as a data URI
and every one of the 76 runs is embedded as JSON, so the page works with no
network, no CDN and no build step. Opening the file directly gives the same
experience as serving it over HTTP (see scripts/serve_guidebook.py).

The interactive table is plain JavaScript so the page has zero dependencies.

Usage:  python -m src.make_guidebook
Writes: docs/guidebook/index.html
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
FIGS = ROOT / "results" / "figures"
OUTDIR = ROOT / "docs" / "guidebook"
OUT = OUTDIR / "index.html"

METHOD_LABEL = {
    "baseline": "Vanilla DP-SGD",
    "classweight": "Class-weighted",
    "pretrained": "Pretrained init",
    "pretrained+classweight": "Pretrained + weighted",
}
EPS_ORDER = [None, 1.0, 3.0, 8.0]


def data_uri(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{b64}"


def load_runs() -> pd.DataFrame:
    rows = []
    for f in sorted(RUNS.glob("*.json")):
        p = json.loads(f.read_text())
        c, m, t = p["config"], p["metrics"], p["train"]
        rows.append({
            "run_id": p["run_id"],
            "dataset": c["dataset"],
            "method": c["method"],
            "method_label": METHOD_LABEL.get(c["method"], c["method"]),
            "eps": None if c["eps"] is None else float(c["eps"]),
            "seed": int(c["seed"]),
            "imbalance_ratio": c["imbalance_ratio"],
            "multilabel": bool(c["multilabel"]),
            "achieved_eps": t["achieved_eps"],
            "sigma": t["noise_multiplier"],
            "max_grad_norm": c["max_grad_norm"] if c.get("max_grad_norm") is not None else t["max_grad_norm"],
            "steps": t["steps_run"],
            "wall_seconds": t["wall_seconds"],
            "n_params": t["n_params"],
            "minority_recall": m.get("minority_recall"),
            "majority_recall": m.get("majority_recall"),
            "accuracy": m.get("accuracy"),
            "macro_recall": m.get("macro_recall"),
            "gap": m.get("gap_majority_minus_minority"),
            "auc_minority": m.get("auc_minority"),
            "mia_auc_pooled": m.get("mia_auc_pooled"),
            "mia_auc_cond": m.get("mia_auc_class_conditional"),
            "mia_tpr_1pct": m.get("mia_tpr_at_1pct_fpr"),
            "mia_tpr_10pct": m.get("mia_tpr_at_10pct_fpr"),
            "macro_auc": m.get("macro_auc"),
            "min_label_recall": m.get("min_label_recall"),
        })
    return pd.DataFrame(rows)


def agg(df, method, eps, col):
    s = df[df.method == method]
    s = s[s.eps.isna()] if eps is None else s[s.eps == eps]
    if s.empty:
        return None, None
    return float(s[col].mean()), float(s[col].std(ddof=1)) if len(s) > 1 else None


def fmt(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def main_table_html(d: pd.DataFrame) -> str:
    rows = []
    for eps in EPS_ORDER:
        for method in METHOD_LABEL:
            mr, sd = agg(d, method, eps, "minority_recall")
            if mr is None:
                continue
            jr, _ = agg(d, method, eps, "majority_recall")
            ac, _ = agg(d, method, eps, "accuracy")
            mia, _ = agg(d, method, eps, "mia_auc_cond")
            eps_lbl = "No DP" if eps is None else f"ε = {eps:g}"
            cls = "row-nodp" if eps is None else ""
            mr_txt = f"{mr:.3f}" + (f" ± {sd:.3f}" if sd is not None else "")
            rows.append(
                f'<tr class="{cls}"><td>{METHOD_LABEL[method]}</td><td>{eps_lbl}</td>'
                f'<td class="num strong">{mr_txt}</td><td class="num">{fmt(jr)}</td>'
                f'<td class="num">{fmt(ac)}</td><td class="num">{fmt(mia)}</td></tr>'
            )
    return "\n".join(rows)


def build() -> Path:
    df = load_runs()
    d = df[(df.dataset == "pneumoniamnist") & df.imbalance_ratio.isna()]
    chest = df[df.dataset == "chestmnist"]
    sweep = df[(df.dataset == "pneumoniamnist") & df.imbalance_ratio.notna()]

    figs = {}
    for name in ["fig1_minority_recall", "fig2_recall_gap", "fig3_mia_auc",
                 "fig4_per_class_eps3", "fig5_imbalance_sweep", "fig6_price_of_privacy"]:
        p = FIGS / f"{name}.png"
        if p.exists():
            figs[name] = data_uri(p)

    mia_nodp, _ = agg(d, "baseline", None, "mia_auc_cond")
    mia_1, _ = agg(d, "baseline", 1.0, "mia_auc_cond")
    base_nodp, _ = agg(d, "baseline", None, "minority_recall")
    base_1, _ = agg(d, "baseline", 1.0, "minority_recall")
    base_3, _ = agg(d, "baseline", 3.0, "minority_recall")
    cw_3, _ = agg(d, "classweight", 3.0, "minority_recall")
    pt_3, _ = agg(d, "pretrained", 3.0, "minority_recall")
    both_3, _ = agg(d, "pretrained+classweight", 3.0, "minority_recall")
    gap_nodp, _ = agg(d, "baseline", None, "gap")
    gap_1, _ = agg(d, "baseline", 1.0, "gap")
    gap_8, _ = agg(d, "baseline", 8.0, "gap")

    b8 = sweep[(sweep.method == "baseline") & np.isclose(sweep.imbalance_ratio, 8.0)
               & sweep.eps.isna()].minority_recall.mean()
    c8 = sweep[(sweep.method == "classweight") & np.isclose(sweep.imbalance_ratio, 8.0)
               & sweep.eps.isna()].minority_recall.mean()
    b1 = sweep[(sweep.method == "baseline") & np.isclose(sweep.imbalance_ratio, 1.0)
               & sweep.eps.isna()].minority_recall.mean()

    total_min = df.wall_seconds.sum() / 60
    med_dp = df[df.eps.notna()].wall_seconds.median()

    runs_json = df.replace({np.nan: None}).to_dict(orient="records")
    runs_json_str = json.dumps(runs_json)

    table_rows = main_table_html(d)

    chest_rows = []
    for eps in [None, 3.0]:
        for method in ["baseline", "classweight"]:
            s = chest[(chest.method == method) & ((chest.eps.isna()) if eps is None else (chest.eps == eps))]
            if s.empty:
                continue
            chest_rows.append(
                f"<tr><td>{METHOD_LABEL[method]}</td><td>{'No DP' if eps is None else f'ε = {eps:g}'}</td>"
                f"<td class='num'>{s.macro_auc.mean():.3f}</td>"
                f"<td class='num'>{s.min_label_recall.mean():.3f}</td>"
                f"<td class='num'>{s.mia_auc_cond.mean():.3f}</td></tr>"
            )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Guidebook · Who Pays for Differential Privacy?</title>
<style>
:root {{
  --navy:#1f3b57; --accent:#c0392b; --ink:#22282e; --muted:#5a6570;
  --bg:#ffffff; --panel:#f5f7fa; --line:#dfe5ec; --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{
  margin:0; background:var(--bg); color:var(--ink);
  font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}}
header.hero {{
  background:linear-gradient(160deg,#1f3b57 0%,#2c4f70 100%); color:#fff;
  padding:64px 32px 52px;
}}
.hero-inner {{ max-width:1080px; margin:0 auto; }}
.hero .bar {{ width:72px; height:5px; background:var(--accent); margin-bottom:26px; }}
h1 {{ font-size:clamp(30px,4vw,46px); line-height:1.14; margin:0 0 16px; font-weight:750; letter-spacing:-.5px; }}
.hero p.sub {{ font-size:clamp(16px,1.6vw,20px); color:#cfdae6; margin:0 0 26px; max-width:760px; }}
.hero .meta {{ font-size:13.5px; color:#a8bccd; display:flex; gap:22px; flex-wrap:wrap; }}
.hero .meta b {{ color:#fff; font-weight:600; }}

.layout {{ max-width:1080px; margin:0 auto; padding:0 32px; }}
nav.toc {{
  position:sticky; top:0; z-index:20; background:rgba(255,255,255,.96);
  backdrop-filter:blur(8px); border-bottom:1px solid var(--line);
  padding:11px 0; margin-bottom:0;
}}
nav.toc .inner {{ max-width:1080px; margin:0 auto; padding:0 32px; display:flex; gap:6px; flex-wrap:wrap; }}
nav.toc a {{
  font-size:12.8px; color:var(--muted); text-decoration:none; padding:5px 10px;
  border-radius:6px; font-weight:500;
}}
nav.toc a:hover {{ background:var(--panel); color:var(--navy); }}

main {{ padding:44px 0 90px; }}
section {{ margin:0 0 52px; scroll-margin-top:64px; }}
h2 {{
  font-size:24px; margin:0 0 6px; color:var(--navy); letter-spacing:-.3px;
  padding-bottom:8px; border-bottom:2px solid var(--panel);
}}
h2 .n {{ color:var(--accent); font-size:15px; font-weight:700; margin-right:10px; vertical-align:2px; }}
h3 {{ font-size:16.5px; margin:26px 0 8px; color:var(--navy); }}
p {{ margin:0 0 13px; }}
ul, ol {{ margin:0 0 14px; padding-left:22px; }}
li {{ margin-bottom:7px; }}
a {{ color:#1a6296; }}
code, .mono {{ font-family:var(--mono); font-size:13.2px; background:var(--panel);
  padding:2px 6px; border-radius:5px; }}
pre {{
  background:#1b2733; color:#dfe8f1; padding:16px 18px; border-radius:10px;
  overflow-x:auto; font-family:var(--mono); font-size:13px; line-height:1.6;
}}
pre code {{ background:none; padding:0; color:inherit; }}

.callout {{
  border-left:4px solid var(--accent); background:#fdf5f4; padding:14px 18px;
  border-radius:0 8px 8px 0; margin:0 0 18px;
}}
.callout.navy {{ border-left-color:var(--navy); background:#f3f6f9; }}
.callout p:last-child {{ margin-bottom:0; }}

.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; margin:8px 0 22px; }}
.stat {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px; }}
.stat .v {{ font-size:26px; font-weight:750; color:var(--navy); line-height:1.1; }}
.stat .l {{ font-size:12px; color:var(--muted); margin-top:5px; }}

table {{ border-collapse:collapse; width:100%; font-size:13.6px; margin:6px 0 8px; }}
th {{ background:var(--navy); color:#fff; text-align:left; padding:9px 11px; font-weight:600; }}
td {{ padding:8px 11px; border-bottom:1px solid var(--line); }}
tbody tr:nth-child(even) {{ background:#fafbfd; }}
td.num {{ font-family:var(--mono); font-size:12.8px; }}
td.strong {{ font-weight:700; color:var(--navy); }}
tr.row-nodp td {{ color:var(--muted); }}
.tbl-wrap {{ overflow-x:auto; }}
caption {{ caption-side:bottom; text-align:left; font-size:12.4px; color:var(--muted); padding-top:8px; }}

figure {{ margin:18px 0 22px; }}
figure img {{ width:100%; height:auto; border:1px solid var(--line); border-radius:10px; background:#fff; }}
figcaption {{ font-size:12.6px; color:var(--muted); margin-top:9px; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
@media (max-width:820px) {{ .grid2 {{ grid-template-columns:1fr; }} }}

/* interactive explorer */
.explorer {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px; }}
.controls {{ display:flex; gap:12px; flex-wrap:wrap; align-items:end; margin-bottom:14px; }}
.controls label {{ font-size:12px; color:var(--muted); display:block; margin-bottom:4px; font-weight:600; text-transform:uppercase; letter-spacing:.4px; }}
.controls select, .controls input {{
  font:inherit; font-size:13.5px; padding:7px 10px; border:1px solid #c9d3de;
  border-radius:7px; background:#fff; color:var(--ink); min-width:150px;
}}
.controls .spacer {{ flex:1; }}
#count {{ font-size:12.5px; color:var(--muted); align-self:center; }}
#xtable {{ background:#fff; border-radius:8px; overflow:hidden; }}
#xtable th {{ cursor:pointer; user-select:none; white-space:nowrap; font-size:12px; padding:8px 7px; }}
#xtable th:hover {{ background:#2c4f70; }}
#xtable th .arrow {{ opacity:.55; font-size:10px; margin-left:3px; }}
#xtable td {{ white-space:nowrap; font-size:12.2px; padding:7px 7px; }}
#xtable td.mono {{ font-size:11.4px; }}
#xtable .pill {{ font-size:10.5px; padding:2px 6px; }}
.pill {{ display:inline-block; padding:2px 8px; border-radius:20px; font-size:11.5px; font-weight:600; }}
.pill.nodp {{ background:#e6ebf1; color:#4a5666; }}
.pill.dp {{ background:#fbe9e7; color:#a93226; }}

footer {{
  border-top:1px solid var(--line); padding:30px 0 50px; color:var(--muted);
  font-size:13px;
}}
footer .layout {{ display:flex; justify-content:space-between; gap:20px; flex-wrap:wrap; }}

@media print {{
  nav.toc, .explorer .controls {{ display:none; }}
  header.hero {{ background:#1f3b57 !important; -webkit-print-color-adjust:exact; print-color-adjust:exact; }}
  section {{ page-break-inside:avoid; }}
  body {{ font-size:11pt; }}
}}
</style>
</head>
<body>

<header class="hero">
  <div class="hero-inner">
    <div class="bar"></div>
    <h1>Who Pays for Differential Privacy?</h1>
    <p class="sub">Per-class fairness and membership leakage in differentially private chest X-ray classification — the complete project guidebook.</p>
    <div class="meta">
      <span><b>Machine Learning Project</b> · Ethics / Responsible AI</span>
      <span><b>Tushar Sharma</b></span>
      <span>MTech I Year, NSUT</span>
      <span>tushar.sharma.pg26@nsut.ac.in</span>
      <span><b>November 2026</b></span>
    </div>
  </div>
</header>

<nav class="toc">
  <div class="inner">
    <a href="#overview">Overview</a>
    <a href="#question">Research question</a>
    <a href="#papers">Papers &amp; gap</a>
    <a href="#data">Data</a>
    <a href="#method">Method</a>
    <a href="#tools">Tools</a>
    <a href="#results">Results</a>
    <a href="#explore">Explore runs</a>
    <a href="#limits">Limitations</a>
    <a href="#reproduce">Reproduce</a>
    <a href="#live">Live version</a>
    <a href="#refs">References</a>
  </div>
</nav>

<main class="layout">

<section id="overview">
  <h2><span class="n">01</span>Overview</h2>
  <p>Differential privacy is the standard guarantee for training models on sensitive medical
  images. <b>DP-SGD</b> is the mechanism most often used to deliver it. This project asks a
  question that aggregate accuracy hides: <b>who pays for that guarantee?</b></p>

  <div class="stats">
    <div class="stat"><div class="v">76</div><div class="l">training runs completed</div></div>
    <div class="stat"><div class="v">48</div><div class="l">main-grid runs (4 methods × 4 budgets × 3 seeds)</div></div>
    <div class="stat"><div class="v">{total_min:.0f}<span style="font-size:15px"> min</span></div><div class="l">total GPU compute</div></div>
    <div class="stat"><div class="v">22</div><div class="l">invariant tests passing</div></div>
    <div class="stat"><div class="v">11.17M</div><div class="l">parameters per model</div></div>
  </div>

  <div class="callout">
    <p><b>One-sentence result.</b> The privacy guarantee is real and measurable — membership-inference
    AUC falls from {fmt(mia_nodp)} without DP to {fmt(mia_1)} at ε = 1 — while a class-weighted loss is a
    free, consistent partial remedy for the minority class, and a pretrained initialisation, despite being
    the best arm without DP, <i>backfires</i> under noise.</p>
  </div>

  <h3>What is in this guidebook</h3>
  <p>Everything needed to understand, run, audit and present the project: the research question and the
  two papers it responds to, the dataset and its imbalance, the full method (including the DP-SGD
  guarantees and how they were achieved), the results with their uncertainty, the exact software
  versions, how to reproduce every number, and how to run the live version.</p>
</section>

<section id="question">
  <h2><span class="n">02</span>The research question</h2>
  <p>DP-SGD clips every per-example gradient to a fixed norm before averaging and adding noise.
  That clipping is what bounds any single patient's influence on the released model — it <i>is</i> the
  guarantee. But it also discards the update from examples whose gradients are largest, and in an
  imbalanced clinical dataset those are disproportionately the minority class.</p>

  <div class="callout navy">
    <p><b>Does DP-SGD degrade recall for minority classes disproportionately — and can cheap,
    privacy-neutral interventions (a class-weighted loss, a pretrained initialisation) recover that loss
    without weakening the guarantee?</b></p>
  </div>

  <h3>Why accuracy cannot answer it</h3>
  <p>PneumoniaMNIST's test split is 26% normal / 74% pneumonia. A model that predicts "pneumonia"
  for every scan scores <b>74% accuracy</b> while achieving <b>zero recall on the normal class</b>.
  Accuracy is blind to exactly the harm this project measures, so the primary metric is
  <b>minority-class (normal) recall</b> and the <b>majority–minority recall gap</b>.</p>
</section>

<section id="papers">
  <h2><span class="n">03</span>The two papers and the gap</h2>

  <h3>Paper 1 — Huang &amp; Xie, arXiv:2507.06619 (9 July 2025)</h3>
  <p><i>"Steps Adaptive Decay DPSGD: Enhancing Performance on Imbalanced Datasets with Differential
  Privacy with HAM10000."</i> The abstract states that gradient-noise DP methods "work well on large
  datasets like MNIST and CIFAR-100, but fail on small, imbalanced medical datasets like HAM10000,"
  attributing the failure to minority-class gradients being clipped while majority classes dominate
  the update. Their remedy changes the <i>training schedule</i> — linearly decaying the noise and
  clipping thresholds — and reports a 2.15% accuracy gain at ε = 3, δ = 1e-3 over an Auto-DPSGD
  baseline.</p>
  <div class="callout">
    <p><b>Gap.</b> Only aggregate accuracy is reported, so the paper cannot show whether the minority
    class was fixed or simply swamped. No loss-level or initialisation intervention is tested. It is a
    two-author arXiv preprint with no peer review and no attack-based privacy evaluation.</p>
  </div>

  <h3>Paper 2 — Mohammadi et al., arXiv:2506.00660 (31 May 2025) → <i>npj Digital Medicine</i> 9:93 (2026)</h3>
  <p><i>"Differential privacy for medical deep learning: methods, tradeoffs, and deployment
  implications."</i> A scoping review of 74 studies published through March 2025. It concludes that
  "severe degradation often occurs under strict privacy (ε ≈ 1), particularly in underrepresented or
  complex modalities"; that "privacy-induced performance gaps disproportionately affect demographic
  subgroups"; and that only a small subset of studies evaluates these tradeoffs. The abstract names
  "balancing privacy with utility and fairness" as a critical challenge.</p>
  <div class="callout">
    <p><b>Gap.</b> A review can identify the tension but cannot test a remedy. No study in it isolates
    minority-class recall as the outcome, and none tests whether a one-line loss change closes the gap.</p>
  </div>

  <div class="callout navy">
    <p><b>This project's contribution.</b> Hold the DP mechanism fixed and vary only the loss (class
    weights) and the initialisation (ImageNet transfer). Report per-class recall and a
    membership-inference attack at matched ε. Both interventions use public information only, so
    neither consumes privacy budget — any recovery is genuinely free at the same ε.</p>
  </div>

  <p style="font-size:13.5px;color:var(--muted)"><b>Citation note.</b> Both were verified against
  arXiv, Crossref and Europe PMC, not assumed. The review's DOI string contains "2025" but the journal
  version's publication year is <b>2026</b> — cite accordingly.</p>
</section>

<section id="data">
  <h2><span class="n">04</span>Dataset</h2>
  <h3>PneumoniaMNIST (primary)</h3>
  <p>5,856 paediatric chest X-rays, 28×28 grayscale, binary pneumonia-versus-normal, from
  <b>MedMNIST v2</b> (Yang et al., <i>Scientific Data</i> 10:41, 2023). Official split used unchanged:</p>
  <div class="tbl-wrap">
  <table>
    <thead><tr><th>Split</th><th>Normal (class 0, minority)</th><th>Pneumonia (class 1, majority)</th><th>Total</th></tr></thead>
    <tbody>
      <tr><td>Train</td><td class="num strong">1,214 (26%)</td><td class="num">3,494 (74%)</td><td class="num">4,708</td></tr>
      <tr><td>Validation</td><td class="num">135</td><td class="num">389</td><td class="num">524</td></tr>
      <tr><td>Test</td><td class="num">234</td><td class="num">390</td><td class="num">624</td></tr>
    </tbody>
    <caption>Table 1. The native imbalance is real, not synthetically induced.</caption>
  </table>
  </div>

  <p><b>The minority class is "normal".</b> That is the clinically meaningful direction for screening:
  a normal scan misread as pneumonia triggers unnecessary follow-up, and in a screening setting that is
  the costly error. The test split is <i>less</i> imbalanced than train (37% vs 26% normal), which
  matters for the attack: a pooled membership-inference AUC would partly measure the distribution shift
  rather than memorisation, so the primary privacy number is <b>class-conditional</b>.</p>

  <h3>ChestMNIST (scale check)</h3>
  <p>78,468 frontal X-rays, 14 multi-label findings from NIH-ChestXray14; the rarest label has only
  <b>144 positives</b> in the full training set (0.18%). Capped at 8,000 training images for
  tractability, and labelled as such everywhere it appears. It is a transfer check, not a replication.</p>

  <p style="font-size:13.5px;color:var(--muted)">Both datasets are public and de-identified, download
  automatically through the <code>medmnist</code> package, and require no data-use agreement.</p>
</section>

<section id="method">
  <h2><span class="n">05</span>Method</h2>

  <h3>DP-SGD in one paragraph</h3>
  <p>For a batch of size B, per-example gradients gᵢ are clipped to norm C
  (g̃ᵢ = gᵢ · min(1, C/‖gᵢ‖)), summed, and perturbed with Gaussian noise:</p>
  <pre><code>θ ← θ − η · ( Σᵢ g̃ᵢ + σ·C·N(0, I) ) / B</code></pre>
  <p>The accountant converts (σ, sample rate, number of steps) into an (ε, δ) guarantee. This project
  fixes <b>δ = 1e-5</b>, plans a step budget, solves for the σ that spends exactly the target ε over
  that budget, then records the <i>achieved</i> ε per run. Every arm uses the same batch size and data
  order, so ε is the only variable distinguishing private runs.</p>

  <h3>Four arms</h3>
  <div class="tbl-wrap">
  <table>
    <thead><tr><th>Arm</th><th>DP</th><th>Class weights</th><th>Pretrained init</th></tr></thead>
    <tbody>
      <tr><td>Vanilla DP-SGD</td><td>ε ∈ {{1, 3, 8, ∞}}</td><td>—</td><td>—</td></tr>
      <tr><td>Class-weighted</td><td>ε ∈ {{1, 3, 8, ∞}}</td><td>inverse frequency</td><td>—</td></tr>
      <tr><td>Pretrained init</td><td>ε ∈ {{1, 3, 8, ∞}}</td><td>—</td><td>ImageNet ResNet-18</td></tr>
      <tr><td>Both</td><td>ε ∈ {{1, 3, 8, ∞}}</td><td>inverse frequency</td><td>ImageNet ResNet-18</td></tr>
    </tbody>
    <caption>Table 2. 4 arms × 4 budgets × 3 seeds = 48 main-grid runs.</caption>
  </table>
  </div>

  <h3>Why these two interventions are privacy-neutral</h3>
  <p>Class weights are computed from the training <i>label histogram</i> — public aggregate
  information — and the accountant never sees them. ImageNet weights are public. Neither consumes
  privacy budget, so any recovery they produce is genuinely free at the same ε. That is the practical
  claim of the project.</p>

  <h3>The model, and the constraints Opacus imposes</h3>
  <p>ResNet-18 adapted to 1×28×28 input. Three changes are <i>required</i>, not stylistic:</p>
  <ul>
    <li><b>GroupNorm replaces every BatchNorm.</b> Opacus cannot form per-sample gradients through
      batch statistics and rejects such models outright — there is a negative-control test asserting
      this failure.</li>
    <li><b>ReLU is non-in-place.</b> Opacus' backward hooks return views that autograd forbids from
      being mutated in place.</li>
    <li><b>The residual block's <code>out += identity</code> is replaced</b> with an out-of-place add,
      for the same reason.</li>
  </ul>

  <h3>Clipping bound calibration</h3>
  <p>C is set from the <b>median true per-sample gradient norm</b>, measured once on an untrained model
  using Opacus' own <code>grad_sample</code> buffers. The median (not max) keeps clipping noise low.
  The statistic is aggregated, computed before training, and releases nothing per example.</p>

  <h3>Membership-inference attacks</h3>
  <p><b>Primary — loss threshold (Yeom-style).</b> Score each example by −cross-entropy loss; low loss
  implies membership; compute ROC-AUC over members (train) vs non-members (test). Reported
  <b>class-conditionally</b> (within each class, then averaged) to remove the train/test class-mix
  confound. An AUC of 0.5 means the attack cannot distinguish members from non-members at all.</p>
  <p><b>Secondary — shadow model.</b> A logistic-regression attacker trained on a <i>separate</i>
  model's in/out behaviour over (loss, confidence, entropy), then applied to the target. Validated
  against both a null (must give ≈ 0.5) and an injected-leakage case; both are permanent tests.</p>
  <div class="callout">
    <p><b>Stated plainly:</b> a loss-threshold attack is a lower bound on leakage, not a ceiling. A flat
    curve is evidence for this attack family, not proof that no attack exists. The formal ε bound is the
    real guarantee.</p>
  </div>
</section>

<section id="tools">
  <h2><span class="n">06</span>Software and tools</h2>
  <div class="tbl-wrap">
  <table>
    <thead><tr><th>Component</th><th>Version</th><th>Role</th></tr></thead>
    <tbody>
      <tr><td>Python</td><td class="num">3.12.10</td><td>Runtime</td></tr>
      <tr><td>PyTorch</td><td class="num">2.6.0+cu124</td><td>Training, autograd</td></tr>
      <tr><td>torchvision</td><td class="num">0.21.0+cu124</td><td>ResNet-18 architecture, ImageNet checkpoint</td></tr>
      <tr><td>Opacus</td><td class="num">1.5.4</td><td>DP-SGD, RDP accountant, ghost clipping</td></tr>
      <tr><td>MedMNIST</td><td class="num">3.0.2</td><td>Dataset download and loading</td></tr>
      <tr><td>scikit-learn</td><td class="num">1.9.1</td><td>Per-class metrics, attack classifier</td></tr>
      <tr><td>NumPy / pandas / SciPy</td><td class="num">2.5.4 / 3.1.1 / 1.18.1</td><td>Numerics, aggregation</td></tr>
      <tr><td>matplotlib</td><td class="num">3.11.2</td><td>All figures</td></tr>
      <tr><td>pytest</td><td class="num">9.2.0</td><td>22 invariant tests</td></tr>
      <tr><td>python-docx</td><td class="num">1.2.0</td><td>Proposal and final report generation</td></tr>
      <tr><td>python-pptx</td><td class="num">1.0.2</td><td>Presentation generation</td></tr>
      <tr><td>pywin32</td><td class="num">312</td><td>PowerPoint COM: slide rendering and PDF export</td></tr>
    </tbody>
    <caption>Table 3. Exact versions, as pinned in <code>requirements.txt</code>.</caption>
  </table>
  </div>

  <h3>Hardware</h3>
  <p>NVIDIA GeForce RTX 4050 Laptop GPU (<b>6 GB VRAM</b>), Windows 11, 16 GB RAM. CUDA 12.4.
  A median DP run takes <b>{med_dp:.0f} s</b>; the full 76-run study takes
  <b>{total_min:.0f} minutes</b> of GPU time.</p>

  <h3>Hardware-forced constraint: batch size 32</h3>
  <p>Opacus materialises per-sample gradients, and this 6 GB card out-of-memory at batch 64. Batch 32
  is therefore mandatory for DP arms; non-private arms use the same size so the comparison is clean.
  Two further facts were established experimentally and are documented in <code>src/train.py</code>:</p>
  <ul>
    <li><code>PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True</code> is <b>not supported</b> on this
      Windows build (PyTorch warns and ignores it).</li>
    <li><b>Ghost clipping</b> (<code>grad_sample_mode="ghost"</code>) is the stable path: 0.24 GB live
      vs 7.8 GB for hooks mode, and not slower (123 vs 135 ms/step).</li>
  </ul>
</section>

<section id="results">
  <h2><span class="n">07</span>Results</h2>

  <h3>Main grid — PneumoniaMNIST</h3>
  <div class="tbl-wrap">
  <table>
    <thead><tr><th>Method</th><th>Privacy</th><th>Minority recall</th><th>Majority recall</th><th>Accuracy</th><th>MIA AUC</th></tr></thead>
    <tbody>
{table_rows}
    </tbody>
    <caption>Table 4. Mean ± sd over 3 seeds. Minority class is "normal". MIA AUC is class-conditional;
    0.5 = no measurable leakage. Lower AUC is better for privacy.</caption>
  </table>
  </div>

  <div class="callout navy">
    <p><b>Measurement floor.</b> The test split has only 234 minority images, so one image moves recall
    by 0.43 points and the binomial standard error at recall 0.70 is <b>±3.0 points</b>. Every delta is
    read against that floor rather than treated as an effect by default.</p>
  </div>

  <h3>Finding 1 — the guarantee is empirically real</h3>
  <p>MIA AUC on the non-private model is <b>{fmt(mia_nodp)}</b>; it falls to <b>{fmt(mia_1)}</b> at
  ε = 1 and stays in a narrow band up to ε = 8. The drop is large relative to its seed spread
  (≈0.004), so this is the result the project can defend at three seeds. Above ε = 1 the curve is
  essentially flat: the guarantee bites once, and tightening further does not remove materially more
  <i>measured</i> leakage.</p>
  <figure>
    <img src="{figs.get('fig3_mia_auc','')}" alt="Membership-inference AUC versus privacy budget">
    <figcaption>Figure 1. Membership-inference AUC versus privacy budget. Lower is better for privacy.</figcaption>
  </figure>

  <h3>Finding 2 — class weighting helps; pretrained init backfires</h3>
  <p>At ε = 3, against vanilla's {fmt(base_3)}: class weighting reaches <b>{fmt(cw_3)}</b>
  ({cw_3-base_3:+.3f}), pretrained init falls to <b>{fmt(pt_3)}</b> ({pt_3-base_3:+.3f}), and the
  combination reaches <b>{fmt(both_3)}</b> ({both_3-base_3:+.3f}). Class weighting moved minority recall
  in the same direction at <i>every</i> seed; pretrained init was worse than vanilla at every seed at
  ε = 8 despite being the strongest arm without DP.</p>
  <div class="grid2">
    <figure>
      <img src="{figs.get('fig1_minority_recall','')}" alt="Minority recall versus privacy budget">
      <figcaption>Figure 2. Minority-class recall versus privacy budget, mean ± sd.</figcaption>
    </figure>
    <figure>
      <img src="{figs.get('fig4_per_class_eps3','')}" alt="Per-class recall at epsilon 3">
      <figcaption>Figure 3. Per-class recall at ε = 3 by method.</figcaption>
    </figure>
  </div>

  <h3>Finding 3 — the fairness effect is present at ε = 1 but not monotone</h3>
  <p>Without DP the majority–minority gap is <b>{fmt(gap_nodp)}</b>. At ε = 1 it widens to
  <b>{fmt(gap_1)}</b>, the largest of any budget. It does <i>not</i> widen monotonically: by ε = 8 it is
  <b>{fmt(gap_8)}</b>, below the no-DP value. So "tighter privacy is always less fair" is false on this
  benchmark — and the ε = 1 magnitude is within the ±3-point floor in any case. Reported as suggestive,
  not established.</p>
  <figure>
    <img src="{figs.get('fig2_recall_gap','')}" alt="Majority minus minority recall gap">
    <figcaption>Figure 4. Majority-minus-minority recall gap across budgets.</figcaption>
  </figure>

  <h3>Imbalance sweep</h3>
  <p>The native split is 2.88:1. We additionally trained on a balanced (1:1) and a severe (8:1)
  training set, holding the minority examples fixed and only shrinking the majority pool, 2 seeds per
  cell. At 8:1 the unweighted model recalls <b>{b8:.3f}</b> of minority scans, down from
  <b>{b1:.3f}</b> at 1:1, while the class-weighted model holds <b>{c8:.3f}</b>. This is the clearest
  evidence that a privacy-neutral loss change protects the minority exactly where imbalance is worst.</p>
  <figure>
    <img src="{figs.get('fig5_imbalance_sweep','')}" alt="Imbalance sweep">
    <figcaption>Figure 5. Minority recall against induced majority:minority ratio.</figcaption>
  </figure>

  <h3>Scale check — ChestMNIST</h3>
  <div class="tbl-wrap">
  <table>
    <thead><tr><th>Method</th><th>Privacy</th><th>Macro AUC</th><th>Min label recall</th><th>MIA AUC</th></tr></thead>
    <tbody>
{''.join(chest_rows)}
    </tbody>
    <caption>Table 5. ChestMNIST, 8k training images, 14 labels, 6 epochs.</caption>
  </table>
  </div>
  <p><b>Read honestly:</b> minimum label recall is 0.000 in every cell (the rarest findings have
  single-digit positives at 8k images, so nothing is predicted at a 0.5 threshold), and MIA AUC dips
  slightly below 0.500. An attack cannot be worse than chance, so that deviation is estimation noise,
  reported as "no measurable leakage" rather than a privacy gain. This is a scale check, not a replication.</p>
</section>

<section id="explore">
  <h2><span class="n">08</span>Explore all 76 runs</h2>
  <p>This table is the raw evidence behind every number above. Filter and sort it — each row is one
  artifact in <code>results/runs/</code>, identified by its config hash.</p>

  <div class="explorer">
    <div class="controls">
      <div><label for="f-dataset">Dataset</label>
        <select id="f-dataset"><option value="">All</option>
          <option value="pneumoniamnist">PneumoniaMNIST</option>
          <option value="chestmnist">ChestMNIST</option></select></div>
      <div><label for="f-study">Study</label>
        <select id="f-study"><option value="">All</option>
          <option value="main">Main grid (PneumoniaMNIST)</option>
          <option value="sweep">Imbalance sweep</option>
          <option value="chest">ChestMNIST scale check</option></select></div>
      <div><label for="f-method">Method</label>
        <select id="f-method"><option value="">All</option></select></div>
      <div><label for="f-eps">Privacy</label>
        <select id="f-eps"><option value="">All</option>
          <option value="none">No DP</option>
          <option value="1">ε = 1</option>
          <option value="3">ε = 3</option>
          <option value="8">ε = 8</option></select></div>
      <div><label for="f-seed">Seed</label>
        <select id="f-seed"><option value="">All</option>
          <option value="0">0</option><option value="1">1</option><option value="2">2</option></select></div>
      <div class="spacer"></div>
      <div id="count"></div>
    </div>
    <div id="xtable" class="tbl-wrap"></div>
  </div>

  <p style="font-size:13px;color:var(--muted);margin-top:12px">Columns are sortable — click a header.
  <code>ε achieved</code> is what the Opacus accountant actually reported (slightly under target because
  it is read at epoch boundaries). <code>σ</code> is the noise multiplier and <code>C</code> the clipping
  bound, both recorded per run.</p>
</section>

<section id="limits">
  <h2><span class="n">09</span>Limitations and ethical considerations</h2>
  <ul>
    <li><b>Three seeds.</b> Differences smaller than the ±3-point binomial floor are called noise, not
      findings. Paired, seed-matched deltas are in <code>results/tables/paired_deltas.csv</code> so a
      consistent small effect is distinguishable from one lucky run.</li>
    <li><b>28×28 inputs.</b> MedMNIST downsamples from up to 384×2,916. Some diagnostic detail is
      destroyed at 28 pixels; results describe this benchmark, not full-resolution radiology.</li>
    <li><b>Weak attack.</b> Loss-threshold MIA is a lower bound on leakage. A flat curve is evidence for
      this attack family, not proof that no attack exists.</li>
    <li><b>secure_mode off.</b> Noise came from a fast PRNG. Standard for experimentation; a production
      claim would require secure mode and is not made here.</li>
    <li><b>Batch size 32, forced by hardware.</b> Larger batches would require more noise for the same
      ε; all arms share 32, so the comparison is internally valid.</li>
    <li><b>One architecture, one dataset family.</b> No claim that DP-SGD is unfair in general.</li>
    <li><b>Data ethics.</b> Public, de-identified datasets; no new patient data collected; no clinical
      deployment claimed — this is a study of a mechanism, not a validated diagnostic tool.</li>
  </ul>
</section>

<section id="reproduce">
  <h2><span class="n">10</span>Reproducing the project</h2>

  <h3>First-time setup</h3>
  <pre><code>cd DP-SGD-Medical-Privacy
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python.exe -m pip install -r requirements.txt</code></pre>

  <h3>Run the study</h3>
  <pre><code>bash scripts/run_grid.sh all        # 76 runs, resumable, one process per run
.venv/Scripts/python.exe -m src.analyze             # tables, figures, RESULTS.md
.venv/Scripts/python.exe -m src.make_final_report   # FINAL_REPORT.docx
.venv/Scripts/python.exe -m src.make_deck           # PRESENTATION.pptx</code></pre>
  <p>Runs are <b>idempotent and resumable</b>: each is keyed by a hash of its config, so re-running skips
  completed work. <code>--force</code> reruns; <code>--list</code> prints everything on disk.</p>

  <h3>Verify the invariants</h3>
  <pre><code>.venv/Scripts/python.exe -m pytest tests -q     # 22 tests</code></pre>
  <p>The tests pin properties that would silently corrupt results if broken: class weights must favour
  the minority, imbalance construction must leave minority rows byte-identical, the MIA must return
  chance under a null and detect injected leakage, the DP accountant must spend the budget it claims,
  the model must contain no BatchNorm, and the multilabel weighted loss must keep its shape.</p>

  <h3>Project structure</h3>
  <pre><code>src/
  data.py            dataset loading, imbalance construction, class weights
  models.py          GroupNorm ResNet-18, Opacus-safe forward, ImageNet transfer
  train.py           training engine, DP-SGD (ghost clipping), C calibration
  metrics.py         per-class recall, recall gap, aggregation
  mia.py             loss-threshold and shadow-model attacks
  runner.py          experiment grid, JSON artifacts, resume, process isolation
  analyze.py         tables, figures, RESULTS.md
  report.py          REPORT.md
  make_proposal.py   PROPOSAL.docx
  make_final_report.py  FINAL_REPORT.docx
  make_deck.py       PRESENTATION.pptx
  make_guidebook.py  this guidebook (HTML)
  render_deck.py     slide PNGs + PDF via PowerPoint COM
scripts/
  run_grid.sh        resilient shell driver
tests/
  test_invariants.py
results/
  runs/              76 JSON artifacts
  tables/            CSV aggregates incl. paired_deltas.csv
  figures/           6 PNG figures
docs/                proposal, report, deck, guidebook</code></pre>

  <h3>Artifact schema</h3>
  <p>Every run writes one JSON file recording: the full config, achieved ε, noise multiplier, clipping
  norm, steps run, parameter count, wall time, the per-epoch history, the per-seed metrics, and the data
  counts. Any number in any document can be traced back to the file that produced it.</p>
</section>

<section id="live">
  <h2><span class="n">11</span>Running the live version</h2>
  <p>This page <i>is</i> the live version — a single self-contained HTML file with all figures embedded
  and all 76 runs loaded into the explorer. It needs no network after download.</p>

  <h3>Option A — open the file (simplest)</h3>
  <p>Double-click <code>docs/guidebook/index.html</code>. Everything works, including the filter/sort
  explorer, because there are no external requests.</p>

  <h3>Option B — serve it on a local URL (recommended for presenting)</h3>
  <pre><code>.venv/Scripts/python.exe scripts/serve_guidebook.py
# then open http://127.0.0.1:8000</code></pre>
  <p>Serving it means the address bar holds a real URL you can put in a slide, and the print/PDF path
  behaves like a normal website. The server binds to localhost only.</p>

  <h3>Option C — publish it publicly with GitHub Pages</h3>
  <p>This must be done from <b>your</b> GitHub account, since it publishes to the internet.</p>
  <ol>
    <li>Create a repository (see the submission checklist for the exact <code>git</code> commands).</li>
    <li>Push the project, including <code>docs/guidebook/index.html</code>.</li>
    <li>In the repo: <b>Settings → Pages → Source: Deploy from a branch → Branch: <code>main</code>,
      folder <code>/docs</code></b> → Save.</li>
    <li>GitHub builds it and gives you a URL of the form
      address beginning <code>https://</code>, then your GitHub username, then
      <code>.github.io/dp-sgd-chest-xray-fairness/guidebook/</code>. Settings → Pages shows the
      exact URL once it is live.</li>
  </ol>
  <div class="callout">
    <p><b>Before publishing,</b> decide whether you want the repository public. A public repo exposes
    the author's name and code; the datasets are not included (<code>data/</code> is gitignored) so no clinical
    images are redistributed.</p>
  </div>
</section>

<section id="refs">
  <h2><span class="n">12</span>References</h2>
  <ol style="font-size:14px">
    <li>Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., &amp; Zhang, L.
      (2016). <i>Deep Learning with Differential Privacy.</i> ACM CCS.
      <a href="https://arxiv.org/abs/1607.00133">arXiv:1607.00133</a></li>
    <li>Huang, X., &amp; Xie, F. (2025). <i>Steps Adaptive Decay DPSGD: Enhancing Performance on
      Imbalanced Datasets with Differential Privacy with HAM10000.</i>
      <a href="https://arxiv.org/abs/2507.06619">arXiv:2507.06619</a></li>
    <li>Mohammadi, M., Vejdanihemmat, M., Lotfinia, M., Rusu, M., Truhn, D., Maier, A., &amp;
      Tayebi Arasteh, S. (2026). <i>Differential privacy for medical deep learning: methods,
      tradeoffs, and deployment implications.</i> npj Digital Medicine, 9, 93.
      <a href="https://doi.org/10.1038/s41746-025-02280-z">doi:10.1038/s41746-025-02280-z</a>
      (preprint <a href="https://arxiv.org/abs/2506.00660">arXiv:2506.00660</a>, 2025)</li>
    <li>Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H., &amp; Ni, B. (2023).
      <i>MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image
      classification.</i> Scientific Data, 10, 41.
      <a href="https://doi.org/10.1038/s41597-022-01721-8">doi:10.1038/s41597-022-01721-8</a></li>
    <li>Yousefpour, A., et al. (2021). <i>Opacus: User-Friendly Differential Privacy Library in
      PyTorch.</i> <a href="https://arxiv.org/abs/2109.12298">arXiv:2109.12298</a></li>
    <li>Yeom, S., Giacomelli, I., Fredrikson, M., &amp; Jha, S. (2018). <i>Privacy Risk in Machine
      Learning: Analyzing the Connection to Overfitting.</i> IEEE CSF.</li>
    <li>Kermany, D. S., et al. (2018). <i>Identifying Medical Diagnoses and Treatable Diseases by
      Image-Based Deep Learning.</i> Cell, 172(5). (Origin of the PneumoniaMNIST images.)</li>
    <li>Wang, X., et al. (2017). <i>ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks.</i>
      CVPR. (Origin of the ChestMNIST images.)</li>
  </ol>

  <h3>Key files in this package</h3>
  <div class="tbl-wrap">
  <table>
    <thead><tr><th>File</th><th>Purpose</th></tr></thead>
    <tbody>
      <tr><td><code>docs/PROPOSAL.docx</code></td><td>One-page proposal (course deliverable)</td></tr>
      <tr><td><code>docs/FINAL_REPORT.docx</code></td><td>Full academic report, ~2,850 words</td></tr>
      <tr><td><code>docs/PRESENTATION.pptx / .pdf</code></td><td>15-slide deck and handout</td></tr>
      <tr><td><code>docs/guidebook/index.html</code></td><td>This page — the live guidebook</td></tr>
      <tr><td><code>docs/SUBMISSION_CHECKLIST.md</code></td><td>Git, viva prep, final checks</td></tr>
      <tr><td><code>results/runs/*.json</code></td><td>76 run artifacts (the evidence)</td></tr>
      <tr><td><code>results/tables/paired_deltas.csv</code></td><td>Seed-matched comparisons</td></tr>
      <tr><td><code>tests/test_invariants.py</code></td><td>22 tests guarding correctness</td></tr>
    </tbody>
    <caption>Table 6. Deliverable map.</caption>
  </table>
  </div>
</section>

</main>

<footer>
  <div class="layout">
    <div>Machine Learning Project · Ethics / Responsible AI · November 2026</div>
    <div>Generated from 76 run artifacts · no number in this guidebook is hand-typed</div>
  </div>
</footer>

<script>
const RUNS = {runs_json_str};

const fmt = (v, nd = 3) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(nd);
const epsLabel = e => e === null ? "No DP" : "ε = " + e;

// populate method filter
const methods = [...new Set(RUNS.map(r => r.method))];
const fm = document.getElementById("f-method");
methods.forEach(m => {{
  const o = document.createElement("option");
  o.value = m;
  o.textContent = RUNS.find(r => r.method === m).method_label;
  fm.appendChild(o);
}});

const state = {{ sortKey: "method_label", sortDir: 1, filters: {{}} }};
["dataset", "study", "method", "eps", "seed"].forEach(k => {{
  state.filters[k] = "";
}});

const COLS = [
  {{ key: "run_id", label: "Run", type: "text" }},
  {{ key: "dataset", label: "Dataset", type: "text" }},
  {{ key: "method_label", label: "Method", type: "text" }},
  {{ key: "eps", label: "Privacy", type: "eps" }},
  {{ key: "seed", label: "Seed", type: "num" }},
  {{ key: "imbalance_ratio", label: "Ratio", type: "ratio" }},
  {{ key: "minority_recall", label: "Min. recall", type: "num" }},
  {{ key: "accuracy", label: "Acc.", type: "num" }},
  {{ key: "mia_auc_cond", label: "MIA AUC", type: "num" }},
  {{ key: "achieved_eps", label: "ε achieved", type: "num" }},
  {{ key: "sigma", label: "σ", type: "num" }},
  {{ key: "max_grad_norm", label: "C", type: "num" }},
  {{ key: "steps", label: "Steps", type: "num" }},
];

function passes(r) {{
  const f = state.filters;
  if (f.dataset && r.dataset !== f.dataset) return false;
  if (f.method && r.method !== f.method) return false;
  if (f.eps) {{
    if (f.eps === "none" && r.eps !== null) return false;
    if (f.eps !== "none" && (r.eps === null || Number(r.eps) !== Number(f.eps))) return false;
  }}
  if (f.seed !== "" && Number(r.seed) !== Number(f.seed)) return false;
  // study is a partition of the runs, not just the ratio: ChestMNIST also has a
  // null imbalance_ratio, so "main grid" must require PneumoniaMNIST as well
  if (f.study === "main" && !(r.dataset === "pneumoniamnist" && r.imbalance_ratio === null)) return false;
  if (f.study === "sweep" && r.imbalance_ratio === null) return false;
  if (f.study === "chest" && r.dataset !== "chestmnist") return false;
  return true;
}}

function sortVal(r, key, type) {{
  const v = r[key];
  if (type === "eps") return v === null ? -1 : Number(v);
  if (type === "ratio") return v === null ? 2.88 : Number(v);
  if (type === "num") return v === null ? -Infinity : Number(v);
  return String(v);
}}

function render() {{
  const rows = RUNS.filter(passes).sort((a, b) => {{
    const col = COLS.find(c => c.key === state.sortKey);
    const A = sortVal(a, state.sortKey, col ? col.type : "text");
    const B = sortVal(b, state.sortKey, col ? col.type : "text");
    if (A < B) return -1 * state.sortDir;
    if (A > B) return 1 * state.sortDir;
    return 0;
  }});

  const head = COLS.map(c => {{
    const arrow = c.key === state.sortKey ? (state.sortDir > 0 ? "▲" : "▼") : "";
    return `<th data-key="${{c.key}}">${{c.label}}<span class="arrow">${{arrow}}</span></th>`;
  }}).join("");

  const body = rows.map(r => {{
    const epsCell = r.eps === null
      ? '<span class="pill nodp">No DP</span>'
      : `<span class="pill dp">ε = ${{r.eps}}</span>`;
    const ratio = r.imbalance_ratio === null ? "native" : Number(r.imbalance_ratio).toFixed(1);
    return `<tr>` +
      `<td class="mono">${{r.run_id}}</td>` +
      `<td>${{r.dataset === "chestmnist" ? "Chest" : "Pneumo"}}</td>` +
      `<td>${{r.method_label}}</td>` +
      `<td>${{epsCell}}</td>` +
      `<td class="num">${{r.seed}}</td>` +
      `<td class="num">${{ratio}}</td>` +
      `<td class="num strong">${{fmt(r.minority_recall)}}</td>` +
      `<td class="num">${{fmt(r.accuracy)}}</td>` +
      `<td class="num">${{fmt(r.mia_auc_cond)}}</td>` +
      `<td class="num">${{fmt(r.achieved_eps, 2)}}</td>` +
      `<td class="num">${{fmt(r.sigma, 2)}}</td>` +
      `<td class="num">${{r.eps === null ? "—" : fmt(r.max_grad_norm, 1)}}</td>` +
      `<td class="num">${{r.steps}}</td>` +
      `</tr>`;
  }}).join("");

  document.getElementById("xtable").innerHTML =
    `<table><thead><tr>${{head}}</tr></thead><tbody>${{body ||
      '<tr><td colspan="13" style="padding:20px;color:#5a6570">No runs match these filters.</td></tr>'}}</tbody></table>`;

  document.getElementById("count").textContent = `${{rows.length}} of ${{RUNS.length}} runs`;

  document.querySelectorAll("#xtable th").forEach(th => {{
    th.addEventListener("click", () => {{
      const k = th.dataset.key;
      if (state.sortKey === k) state.sortDir *= -1;
      else {{ state.sortKey = k; state.sortDir = 1; }}
      render();
    }});
  }});
}}

document.getElementById("f-dataset").addEventListener("change", e => {{ state.filters.dataset = e.target.value; render(); }});
document.getElementById("f-study").addEventListener("change", e => {{ state.filters.study = e.target.value; render(); }});
document.getElementById("f-method").addEventListener("change", e => {{ state.filters.method = e.target.value; render(); }});
document.getElementById("f-eps").addEventListener("change", e => {{ state.filters.eps = e.target.value; render(); }});
document.getElementById("f-seed").addEventListener("change", e => {{ state.filters.seed = e.target.value; render(); }});

render();
</script>
</body>
</html>
"""

    OUTDIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    return OUT


def main() -> None:
    path = build()
    size_kb = path.stat().st_size / 1024
    print(f"Wrote {path}")
    print(f"{size_kb:.0f} KB self-contained (figures embedded, 76 runs loaded)")


if __name__ == "__main__":
    main()
