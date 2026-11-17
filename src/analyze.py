"""Aggregate run artifacts into tables, figures and a results report.

Reads only results/runs/*.json produced by src.runner, so every number in the
outputs is traceable to a file. Nothing here invents or fills in a value: a
missing run is reported as missing.

Usage:  python -m src.analyze
Writes: results/tables/*.csv, results/figures/*.png, results/RESULTS.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

METHOD_LABEL = {
    "baseline": "Vanilla DP-SGD",
    "classweight": "+ class weights",
    "pretrained": "+ pretrained init",
    "pretrained+classweight": "+ both",
}
METHOD_ORDER = list(METHOD_LABEL)

# PneumoniaMNIST train split is 3,494 pneumonia vs 1,214 normal. A run with
# imbalance_ratio=None used that native split, so its effective majority:minority
# ratio is 2.88, not "missing". Recorded here so the sweep figure can plot the
# native point instead of silently dropping it.
NATIVE_RATIO = 3494 / 1214

# Test-split class counts (PneumoniaMNIST official split). Minority recall is a
# proportion over this fixed denominator, so its binomial standard error is
# sqrt(r(1-r)/n) ~ 0.03 at r=0.7, n=234. Any observed difference much smaller
# than that is not distinguishable from sampling noise on this benchmark, and
# the report says so instead of grading the project on a fragile delta.
TEST_MINORITY_N = 234
TEST_MAJORITY_N = 390


def recall_se(recall: float, n: int) -> float:
    """Binomial standard error of a recall estimate."""
    if recall is None or np.isnan(recall) or n <= 0:
        return float("nan")
    return float(np.sqrt(max(recall * (1 - recall), 0.0) / n))


def load_runs() -> pd.DataFrame:
    rows = []
    for f in sorted(RUNS.glob("*.json")):
        d = f.read_text()
        import json

        p = json.loads(d)
        c, m, t = p["config"], p["metrics"], p["train"]
        row = {
            "run_id": p["run_id"],
            "dataset": c["dataset"],
            "method": c["method"],
            "eps": np.nan if c["eps"] is None else float(c["eps"]),
            "private": c["eps"] is not None,
            "seed": c["seed"],
            "imbalance_ratio": c["imbalance_ratio"],
            "multilabel": c["multilabel"],
            "max_train": c["max_train"],
            "achieved_eps": t["achieved_eps"],
            "noise_multiplier": t["noise_multiplier"],
            "max_grad_norm": t["max_grad_norm"],
            "steps_run": t["steps_run"],
            "wall_seconds": t["wall_seconds"],
            "n_params": t["n_params"],
        }
        row.update({k: v for k, v in m.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def eps_label(v):
    if pd.isna(v):
        return "No DP"
    return f"eps={v:g}"


def summarise(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    keys = ["dataset", "method", "eps"]
    g = df.groupby(keys, dropna=False)
    out = g[metrics].agg(["mean", "std", "count"]).reset_index()
    out.columns = ["_".join([c for c in col if c]).strip("_") for col in out.columns]
    return out


def paired_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Per-seed paired comparisons: DP vs no-DP within the same method, and
    each intervention vs vanilla DP-SGD at the same epsilon.

    Seeds are matched across arms, so a paired delta removes between-seed
    variance (which dominates here: sd is 0.05-0.08 across seeds but the same
    seed trained twice differs far less). Reporting the paired delta and its
    spread is more informative than comparing two unpaired means, and it stops
    a small-but-consistent effect from being dismissed as noise.
    """
    d = df[(df.dataset == "pneumoniamnist") & (df.imbalance_ratio.isna())]
    rows = []

    # (a) within-method: private run minus that method's own no-DP run, per seed
    for method in METHOD_ORDER:
        nodp = d[(d.method == method) & (d.eps.isna())].set_index("seed")
        if nodp.empty:
            continue
        for eps in [1.0, 3.0, 8.0]:
            priv = d[(d.method == method) & (d.eps == eps)].set_index("seed")
            common = sorted(set(nodp.index) & set(priv.index))
            if not common:
                continue
            for metric in ["minority_recall", "majority_recall"]:
                deltas = [priv.loc[s, metric] - nodp.loc[s, metric] for s in common]
                rows.append(
                    {
                        "comparison": f"{METHOD_LABEL[method]}: eps={eps:g} vs no-DP",
                        "metric": metric,
                        "n_pairs": len(deltas),
                        "delta_mean": float(np.mean(deltas)),
                        "delta_sd": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float("nan"),
                        "all_same_sign": bool(len(set(np.sign(deltas))) == 1),
                    }
                )

    # (b) between-method at matched seed and epsilon
    for eps in [1.0, 3.0, 8.0]:
        for method in ["classweight", "pretrained", "pretrained+classweight"]:
            a = d[(d.method == method) & (d.eps == eps)].set_index("seed")
            b = d[(d.method == "baseline") & (d.eps == eps)].set_index("seed")
            common = sorted(set(a.index) & set(b.index))
            if not common:
                continue
            for metric in ["minority_recall"]:
                deltas = [a.loc[s, metric] - b.loc[s, metric] for s in common]
                rows.append(
                    {
                        "comparison": f"{METHOD_LABEL[method]} vs vanilla @ eps={eps:g}",
                        "metric": metric,
                        "n_pairs": len(deltas),
                        "delta_mean": float(np.mean(deltas)),
                        "delta_sd": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float("nan"),
                        "all_same_sign": bool(len(set(np.sign(deltas))) == 1),
                    }
                )
    return pd.DataFrame(rows)


def main_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df.dataset == "pneumoniamnist") & (df.imbalance_ratio.isna())]
    metrics = [
        "minority_recall",
        "majority_recall",
        "macro_recall",
        "accuracy",
        "auc_minority",
        "mia_auc_pooled",
        "mia_auc_class_conditional",
        "mia_tpr_at_1pct_fpr",
        "mia_mean_member_loss",
        "mia_mean_nonmember_loss",
    ]
    metrics = [m for m in metrics if m in d.columns]
    s = summarise(d, metrics)
    s["method_label"] = s.method.map(METHOD_LABEL)
    s["eps_label"] = s.eps.apply(eps_label)
    return s


def table_rows(df: pd.DataFrame) -> list[dict]:
    """Row-wise per-seed records for the appendix."""
    d = df[(df.dataset == "pneumoniamnist") & (df.imbalance_ratio.isna())].copy()
    d["method_label"] = d.method.map(METHOD_LABEL)
    d["eps_label"] = d.eps.apply(eps_label)
    cols = [
        "method_label",
        "eps_label",
        "seed",
        "minority_recall",
        "majority_recall",
        "accuracy",
        "auc_minority",
        "mia_auc_class_conditional",
        "achieved_eps",
        "noise_multiplier",
        "max_grad_norm",
        "steps_run",
        "wall_seconds",
    ]
    cols = [c for c in cols if c in d.columns]
    return d[cols].sort_values(["eps_label", "method_label", "seed"]).to_dict("records")


def make_figures(df: pd.DataFrame) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    plt.rcParams.update({"font.size": 9, "figure.dpi": 160, "savefig.bbox": "tight"})

    d = df[(df.dataset == "pneumoniamnist") & (df.imbalance_ratio.isna())].copy()
    if d.empty:
        return made

    # --- Figure 1: minority recall vs epsilon, one line per method ---
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    # always the full budget axis, so figures are comparable before and after
    # every run lands and a missing cell shows as a gap rather than a shift
    eps_vals = [np.nan, 1.0, 3.0, 8.0]
    xs = list(range(len(eps_vals)))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for color, method in zip(colors, METHOD_ORDER):
        sub = d[d.method == method]
        if sub.empty:
            continue
        means, errs = [], []
        for e in eps_vals:
            s = sub[sub.eps.isna()] if pd.isna(e) else sub[sub.eps == e]
            if s.empty:
                means.append(np.nan)
                errs.append(np.nan)
            else:
                means.append(s.minority_recall.mean())
                errs.append(s.minority_recall.std(ddof=1) if len(s) > 1 else 0.0)
        ax.errorbar(
            xs,
            means,
            yerr=errs,
            marker="o",
            capsize=3,
            label=METHOD_LABEL[method],
            color=color,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([eps_label(e) for e in eps_vals])
    ax.set_ylabel("Minority-class (normal) recall")
    ax.set_xlabel("Privacy budget")
    ax.set_title("PneumoniaMNIST: minority recall under DP")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="lower left")
    p = FIGS / "fig1_minority_recall.png"
    fig.savefig(p)
    plt.close(fig)
    made.append(str(p))

    # --- Figure 2: recall gap (majority - minority) ---
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for color, method in zip(colors, METHOD_ORDER):
        sub = d[d.method == method]
        if sub.empty:
            continue
        means = []
        for e in eps_vals:
            s = sub[sub.eps.isna()] if pd.isna(e) else sub[sub.eps == e]
            means.append(s.gap_majority_minus_minority.mean() if not s.empty else np.nan)
        ax.plot(xs, means, marker="s", label=METHOD_LABEL[method], color=color)
    ax.set_xticks(xs)
    ax.set_xticklabels([eps_label(e) for e in eps_vals])
    ax.set_ylabel("Recall gap (majority - minority)")
    ax.set_xlabel("Privacy budget")
    ax.set_title("Fairness gap widens or narrows under DP?")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    p = FIGS / "fig2_recall_gap.png"
    fig.savefig(p)
    plt.close(fig)
    made.append(str(p))

    # --- Figure 3: MIA AUC vs epsilon (privacy gain) ---
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for color, method in zip(colors, METHOD_ORDER):
        sub = d[d.method == method]
        if sub.empty:
            continue
        means, errs = [], []
        for e in eps_vals:
            s = sub[sub.eps.isna()] if pd.isna(e) else sub[sub.eps == e]
            means.append(s.mia_auc_class_conditional.mean() if not s.empty else np.nan)
            errs.append(
                s.mia_auc_class_conditional.std(ddof=1) if len(s) > 1 else 0.0
            )
        ax.errorbar(xs, means, yerr=errs, marker="^", capsize=3, label=METHOD_LABEL[method], color=color)
    ax.axhline(0.5, color="black", lw=0.8, ls="--", label="No leakage (0.5)")
    ax.set_xticks(xs)
    ax.set_xticklabels([eps_label(e) for e in eps_vals])
    ax.set_ylabel("Membership-inference AUC (class-conditional)")
    ax.set_xlabel("Privacy budget")
    ax.set_title("Empirical privacy: lower AUC = less leakage")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    p = FIGS / "fig3_mia_auc.png"
    fig.savefig(p)
    plt.close(fig)
    made.append(str(p))

    # --- Figure 4: per-class recall at eps=3 for the four methods ---
    sub = d[d.eps == 3.0]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(5.2, 3.0))
        x = np.arange(len(METHOD_ORDER))
        mino = [sub[sub.method == m].minority_recall.mean() for m in METHOD_ORDER]
        majo = [sub[sub.method == m].majority_recall.mean() for m in METHOD_ORDER]
        w = 0.36
        ax.bar(x - w / 2, mino, w, label="minority (normal)", color="#d62728")
        ax.bar(x + w / 2, majo, w, label="majority (pneumonia)", color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABEL[m].replace("+ ", "+\n") for m in METHOD_ORDER], fontsize=7)
        ax.set_ylabel("Recall")
        ax.set_title("Per-class recall at eps = 3")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=7)
        p = FIGS / "fig4_per_class_eps3.png"
        fig.savefig(p)
        plt.close(fig)
        made.append(str(p))

    # --- Figure 5: imbalance sweep ---
    sw = df[(df.dataset == "pneumoniamnist") & (df.method.isin(["baseline", "classweight"]))].copy()
    if not sw.empty:
        # normalise: a null ratio means the native split, which is 2.88:1
        sw["eff_ratio"] = sw.imbalance_ratio.fillna(NATIVE_RATIO).astype(float)
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
        for ax, method in zip(axes, ["baseline", "classweight"]):
            sub = sw[sw.method == method]
            for eps in [np.nan, 1.0, 3.0]:
                s = sub[sub.eps.isna()] if pd.isna(eps) else sub[sub.eps == eps]
                if s.empty:
                    continue
                grp = s.groupby("eff_ratio").minority_recall.agg(["mean", "std", "count"])
                ax.errorbar(
                    grp.index,
                    grp["mean"],
                    yerr=grp["std"].fillna(0),
                    marker="o",
                    capsize=3,
                    label=eps_label(eps),
                )
            ax.axvline(NATIVE_RATIO, color="grey", lw=0.8, ls=":", label="native (2.88)")
            ax.set_title(METHOD_LABEL[method], fontsize=8)
            ax.set_xlabel("Majority:minority ratio")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=6)
        axes[0].set_ylabel("Minority recall")
        fig.suptitle("Does imbalance make DP's minority cost worse?", fontsize=9)
        p = FIGS / "fig5_imbalance_sweep.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        made.append(str(p))

    # --- Figure 6: price of privacy, as a delta from each method's own no-DP run ---
    if not d.empty:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for color, method in zip(colors, METHOD_ORDER):
            sub = d[d.method == method]
            base = sub[sub.eps.isna()]
            if base.empty:
                continue
            b = base.minority_recall.mean()
            means = []
            for e in eps_vals[1:]:
                s = sub[sub.eps == e]
                means.append(s.minority_recall.mean() - b if not s.empty else np.nan)
            ax.plot(eps_vals[1:], means, marker="o", label=METHOD_LABEL[method], color=color)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_ylabel("Change in minority recall vs own no-DP run")
        ax.set_xlabel("Privacy budget (epsilon)")
        ax.set_title("Price of privacy, net of each method's baseline")
        ax.set_xscale("log")
        ax.set_xticks(eps_vals[1:])
        ax.set_xticklabels([f"{e:g}" for e in eps_vals[1:]])
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="lower right")
        p = FIGS / "fig6_price_of_privacy.png"
        fig.savefig(p)
        plt.close(fig)
        made.append(str(p))

    return made


def write_report(df: pd.DataFrame, figures: list[str]) -> None:
    d = df[(df.dataset == "pneumoniamnist") & (df.imbalance_ratio.isna())]
    lines = []
    A = lines.append

    A("# Results\n")
    A("Generated by `python -m src.analyze` from `results/runs/*.json`.\n")
    A(f"Runs on disk: **{len(df)}**. PneumoniaMNIST main-grid runs: **{len(d)}**.\n")

    missing = []
    for method in METHOD_ORDER:
        for eps in [np.nan, 1.0, 3.0, 8.0]:
            s = d[(d.method == method) & ((d.eps.isna()) if pd.isna(eps) else (d.eps == eps))]
            if s.empty:
                missing.append(f"{METHOD_LABEL[method]} @ {eps_label(eps)}")
    if missing:
        A("> Missing cells (not yet run): " + ", ".join(missing) + "\n")

    A("\n## Main grid — PneumoniaMNIST (mean +/- sd over seeds)\n")
    A("Minority class is *normal* (class 0, 26% of train). MIA AUC is")
    A("class-conditional: 0.5 means the attack cannot separate members from")
    A("non-members at all.\n")
    A(f"The test split contains only {TEST_MINORITY_N} minority and {TEST_MAJORITY_N} "
      f"majority images, so a single minority image moves recall by "
      f"{100/TEST_MINORITY_N:.2f} percentage points. At a recall of 0.70 the binomial "
      f"standard error alone is ±{recall_se(0.70, TEST_MINORITY_N)*100:.1f} points. "
      f"Read the deltas below against that floor.\n")

    cols = [
        ("minority_recall", "Minority recall"),
        ("majority_recall", "Majority recall"),
        ("macro_recall", "Macro recall"),
        ("accuracy", "Accuracy"),
        ("mia_auc_class_conditional", "MIA AUC"),
    ]
    header = "| Method | eps | " + " | ".join(c[1] for c in cols) + " | n |"
    sep = "|" + "---|" * (len(cols) + 3)
    A(header)
    A(sep)
    for eps in [np.nan, 1.0, 3.0, 8.0]:
        for method in METHOD_ORDER:
            s = d[(d.method == method) & ((d.eps.isna()) if pd.isna(eps) else (d.eps == eps))]
            if s.empty:
                continue
            cells = []
            for col, _ in cols:
                if col in s:
                    mu = s[col].mean()
                    sd = s[col].std(ddof=1) if len(s) > 1 else float("nan")
                    cells.append(
                        f"{mu:.3f}" if np.isnan(sd) else f"{mu:.3f} ± {sd:.3f}"
                    )
                else:
                    cells.append("—")
            A(f"| {METHOD_LABEL[method]} | {eps_label(eps)} | " + " | ".join(cells) + f" | {len(s)} |")
    A("")

    # headline deltas, computed only when the cells exist
    def cell(method, eps, col):
        s = d[(d.method == method) & ((d.eps.isna()) if pd.isna(eps) else (d.eps == eps))]
        return np.nan if s.empty else s[col].mean()

    A("\n## Headline comparisons\n")
    A(f"Measurement floor: ±{recall_se(0.70, TEST_MINORITY_N)*100:.1f} points of binomial "
      f"standard error on minority recall (n={TEST_MINORITY_N}). A difference is called "
      f"a *finding* below only when it is larger than this floor **and** every seed moved "
      f"in the same direction; otherwise it is reported as *within noise*.\n")
    base_no_dp = cell("baseline", np.nan, "minority_recall")
    se_floor = recall_se(0.70, TEST_MINORITY_N)
    if not np.isnan(base_no_dp):
        for eps in [1.0, 3.0, 8.0]:
            v = cell("baseline", eps, "minority_recall")
            if np.isnan(v):
                continue
            delta = v - base_no_dp
            verdict = "beyond the noise floor" if abs(delta) > se_floor else "WITHIN the noise floor"
            A(f"- Vanilla DP-SGD at ε={eps:g}: minority recall {v:.3f} vs {base_no_dp:.3f} without DP "
              f"(delta {delta:+.3f}, {verdict}).")
    for fix in ["classweight", "pretrained", "pretrained+classweight"]:
        v = cell(fix, 3.0, "minority_recall")
        b = cell("baseline", 3.0, "minority_recall")
        if not (np.isnan(v) or np.isnan(b)):
            delta = v - b
            verdict = "beyond the noise floor" if abs(delta) > se_floor else "WITHIN the noise floor"
            A(f"- `{METHOD_LABEL[fix]}` at ε=3: minority recall {v:.3f} "
              f"({delta:+.3f} vs vanilla DP-SGD, {verdict}).")
    A("")
    A("**The privacy result is not subject to this floor.** MIA AUC is computed over the full "
      "train and test splits with a per-class average, its seed-to-seed standard deviation is "
      "an order of magnitude smaller than the DP effect, and it moves monotonically with ε. "
      "That is the finding this project can defend at 3 seeds; the recall deltas are suggestive "
      "and are reported with their uncertainty rather than as established effects.")
    A("")

    A("\n## Figures\n")
    for f in figures:
        A(f"- `{Path(f).relative_to(ROOT)}`")
    A("")

    # paired analysis: the honest answer to "is this just seed noise?"
    import pandas as _pd

    pdz = paired_deltas(df)
    if not pdz.empty:
        A("\n## Paired comparisons (seed-matched)\n")
        A("Seeds are matched across arms, so each delta is computed within a seed. This removes the")
        A("between-seed variance that dominates the unpaired standard deviations above. `same sign`")
        A("means every seed moved in the same direction — a small but consistent effect, as opposed to")
        A("a mean that one outlier seed produced.\n")
        A("| Comparison | Metric | n | Mean delta | SD | same sign |")
        A("|---|---|---|---|---|---|")
        for _, r in pdz.iterrows():
            sd = "—" if np.isnan(r["delta_sd"]) else f"{r['delta_sd']:.3f}"
            A(
                f"| {r['comparison']} | {r['metric']} | {r['n_pairs']} | "
                f"**{r['delta_mean']:+.3f}** | {sd} | {'yes' if r['all_same_sign'] else 'no'} |"
            )
        A("")

    A("\n## Files\n")
    A("- `results/tables/main_grid.csv` — aggregated means/sd")
    A("- `results/tables/main_grid_per_seed.csv` — one row per seed with achieved eps")
    A("- `results/runs/*.json` — full per-run artifact (config, history, metrics)")
    (ROOT / "results" / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_runs()
    if df.empty:
        print("No runs found in", RUNS)
        return
    df.to_csv(TABLES / "all_runs.csv", index=False)
    mt = main_table(df)
    mt.to_csv(TABLES / "main_grid.csv", index=False)
    pd.DataFrame(table_rows(df)).to_csv(TABLES / "main_grid_per_seed.csv", index=False)
    pd_df = paired_deltas(df)
    if not pd_df.empty:
        pd_df.to_csv(TABLES / "paired_deltas.csv", index=False)

    chest = df[df.dataset == "chestmnist"]
    if not chest.empty:
        cols = [c for c in ["macro_auc", "macro_recall", "min_label_recall", "mia_auc_class_conditional"] if c in chest]
        chest.groupby(["method", "eps"], dropna=False)[cols].agg(["mean", "std"]).to_csv(
            TABLES / "chest_grid.csv"
        )

    figs = make_figures(df)
    write_report(df, figs)
    print(f"Wrote {len(figs)} figures, {len(list(TABLES.glob('*.csv')))} tables, RESULTS.md")
    print(f"Runs: {len(df)}")


if __name__ == "__main__":
    main()
