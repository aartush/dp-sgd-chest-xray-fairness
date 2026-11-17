"""Experiment runner.

Every run writes one self-describing JSON artifact to results/runs/, named by a
stable config hash, so a rerun skips work that already exists and any number in
the paper can be traced back to the file that produced it.

Usage
    python -m src.runner --grid main        # eps x method x seed on PneumoniaMNIST
    python -m src.runner --grid imbalance   # imbalance ratio sweep
    python -m src.runner --grid chest       # ChestMNIST scale check
    python -m src.runner --grid all
    python -m src.runner --list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .data import (
    load_pneumoniamnist,
    load_chestmnist,
    make_imbalanced_train,
    binary_class_counts,
)
from .train import TrainConfig, train
from .metrics import binary_report, multilabel_report
from .mia import per_example_loss, loss_threshold_attack

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
RUNS.mkdir(parents=True, exist_ok=True)

MAIN_METHODS = ["baseline", "classweight", "pretrained", "pretrained+classweight"]
EPSILONS = [None, 1.0, 3.0, 8.0]
SEEDS = [0, 1, 2]


def run_id(cfg: TrainConfig) -> str:
    key = json.dumps(
        {k: v for k, v in cfg.__dict__.items() if k not in ("device",)},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def evaluate(cfg: TrainConfig, result) -> dict:
    m = result.metrics
    out = {}
    if cfg.multilabel:
        probs = 1 / (1 + np.exp(-m["_test_logits"]))
        out.update(multilabel_report(m["_test_labels"], probs))
        train_loss = per_example_loss(m["_train_logits"], m["_train_labels"], multilabel=True)
        test_loss = per_example_loss(m["_test_logits"], m["_test_labels"], multilabel=True)
        out.update(
            loss_threshold_attack(
                train_loss, m["_train_labels"], test_loss, m["_test_labels"], multilabel=True
            )
        )
    else:
        probs = torch.softmax(torch.from_numpy(m["_test_logits"]), 1)[:, 1].numpy()
        out.update(binary_report(m["_test_labels"], probs))
        train_loss = per_example_loss(m["_train_logits"], m["_train_labels"])
        test_loss = per_example_loss(m["_test_logits"], m["_test_labels"])
        out.update(loss_threshold_attack(train_loss, m["_train_labels"], test_loss, m["_test_labels"]))
    return out


def _execute(cfg: TrainConfig, splits) -> dict:
    """Train + evaluate in the current process and write the artifact."""
    rid = run_id(cfg)
    path = RUNS / f"{rid}.json"
    t0 = time.time()
    verbose = os.environ.get("RUNNER_VERBOSE", "") not in ("", "0", "false")
    result = train(cfg, splits, log_fn=print if verbose else (lambda s: None))
    metrics = evaluate(cfg, result)
    payload = {
        "run_id": rid,
        "config": {k: v for k, v in cfg.__dict__.items()},
        "train": {
            "achieved_eps": result.achieved_eps,
            "noise_multiplier": result.noise_multiplier,
            "max_grad_norm": cfg.max_grad_norm,
            "steps_run": result.steps_run,
            "n_params": result.n_params,
            "wall_seconds": round(result.wall_seconds, 2),
            "history": result.history,
        },
        "metrics": {k: v for k, v in metrics.items() if not k.startswith("_")},
        "data": {
            "train_n": len(splits["train"]),
            "val_n": len(splits["val"]),
            "test_n": len(splits["test"]),
            "train_counts": binary_class_counts(splits["train"].labels) if not cfg.multilabel else None,
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    m = payload["metrics"]
    if cfg.multilabel:
        print(f"        macro_auc={m['macro_auc']:.4f} min_label_recall={m['min_label_recall']:.4f}")
    else:
        print(
            f"        minority_recall={m['minority_recall']:.4f} "
            f"majority_recall={m['majority_recall']:.4f} acc={m['accuracy']:.4f} "
            f"mia_cond={m['mia_auc_class_conditional']:.4f}"
        )
    return payload


def run_one(cfg: TrainConfig, splits, force: bool = False) -> dict | None:
    """Run one config, resuming from disk when the artifact already exists.

    Each config is executed by `_execute` in the *current* interpreter only when
    invoked as a single run (`--single`). The grid functions below call
    `run_one`, which spawns a fresh interpreter per run: Opacus' grad-sample
    hooks and CUDA allocator state are not fully released between runs, and a
    shared process died with CUDA OOM on the second DP run even though the GPU
    was empty. A fresh interpreter costs a few seconds of import time and
    removes an entire class of cross-run failures.
    """
    rid = run_id(cfg)
    path = RUNS / f"{rid}.json"
    if path.exists() and not force:
        print(f"  [skip] {rid} exists", flush=True)
        return json.loads(path.read_text())

    print(
        f"  [run ] {rid} {cfg.dataset} {cfg.method} eps={cfg.eps} seed={cfg.seed} "
        f"imb={cfg.imbalance_ratio} max_train={cfg.max_train}",
        flush=True,
    )
    cmd = [
        sys.executable,
        "-m",
        "src.runner",
        "--single",
        json.dumps({k: v for k, v in cfg.__dict__.items()}),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    # surface the child's output (warnings included) but keep the artifact as truth
    if proc.stdout:
        for line in proc.stdout.strip().splitlines():
            if line.strip():
                print("        " + line.strip(), flush=True)
    if proc.returncode != 0:
        print(f"  [FAIL] {rid} (exit {proc.returncode}, {time.time()-t0:.0f}s)")
        tail = (proc.stderr or "").strip().splitlines()[-12:]
        for line in tail:
            print("        " + line)
        return None
    if not path.exists():
        print(f"  [FAIL] {rid} exited 0 but wrote no artifact")
        return None
    return json.loads(path.read_text())


def _load_splits_for(cfg: TrainConfig):
    """Rebuild the exact splits a config was specified against.

    The child process must see byte-identical data to what the parent would have
    built, so the imbalance construction is replayed deterministically from the
    recorded ratio and seed rather than serialised.
    """
    if cfg.dataset == "chestmnist":
        return load_chestmnist(max_train=cfg.max_train, seed=0)
    base = load_pneumoniamnist()
    if cfg.imbalance_ratio is None:
        return base
    tr = make_imbalanced_train(base["train"], ratio=cfg.imbalance_ratio, minority_class=0, seed=0)
    return {"train": tr, "val": base["val"], "test": base["test"]}


def grid_main(force: bool, seeds=None, methods=None, epsilons=None) -> None:
    splits = load_pneumoniamnist()
    seeds = seeds or SEEDS
    methods = methods or MAIN_METHODS
    epsilons = epsilons if epsilons is not None else EPSILONS
    print(f"MAIN GRID: {len(methods)} methods x {len(epsilons)} eps x {len(seeds)} seeds")
    for method in methods:
        for eps in epsilons:
            for seed in seeds:
                cfg = TrainConfig(
                    dataset="pneumoniamnist",
                    method=method,
                    eps=eps,
                    seed=seed,
                    epochs=12,
                    batch_size=32,
                    lr=1e-3,
                    weight_decay=1e-4,
                )
                run_one(cfg, splits, force)


def grid_imbalance(force: bool) -> None:
    """Sweep the majority:minority ratio to show where DP-SGD hurts most.

    The full PneumoniaMNIST train set is 1:2.88 (ratio expressed as
    majority/minority = 2.88). We compare the native ratio against artificially
    balanced (1.0) and strongly imbalanced (8.0) versions to separate the effect
    of imbalance from the effect of dataset size.
    """
    base = load_pneumoniamnist()
    print("IMBALANCE SWEEP: ratios 1.0 / 2.88(native) / 8.0 x method x eps")
    for ratio in [1.0, None, 8.0]:
        if ratio is None:
            splits = base
        else:
            tr = make_imbalanced_train(base["train"], ratio=ratio, minority_class=0, seed=0)
            splits = {"train": tr, "val": base["val"], "test": base["test"]}
        for method in ["baseline", "classweight"]:
            for eps in [None, 1.0, 3.0]:
                for seed in [0, 1]:
                    cfg = TrainConfig(
                        dataset="pneumoniamnist",
                        method=method,
                        eps=eps,
                        seed=seed,
                        epochs=12,
                        batch_size=32,
                        lr=1e-3,
                        imbalance_ratio=ratio,
                    )
                    run_one(cfg, splits, force)


def grid_chest(force: bool, max_train: int = 8000) -> None:
    """Scale check on ChestMNIST multi-label.

    Full ChestMNIST (78k x 14 labels) would make 12 epochs of DP-SGD take hours
    on a laptop GPU; we cap training at 8,000 images, which still keeps the
    rarest label (144 positives in 78k) represented in single digits. That
    scarcity is exactly the regime the paper's gap is about, and the run is
    labelled accordingly rather than presented as a full-data result.
    """
    splits = load_chestmnist(max_train=max_train, seed=0)
    counts = splits["train"].labels.sum(axis=0).astype(int).tolist()
    print(f"CHESTMNIST: train={len(splits['train'])} per-label positives={counts}")
    for method in ["baseline", "classweight"]:
        for eps in [None, 3.0]:
            cfg = TrainConfig(
                dataset="chestmnist",
                method=method,
                eps=eps,
                seed=0,
                epochs=6,
                batch_size=32,
                lr=1e-3,
                multilabel=True,
                max_train=max_train,
            )
            run_one(cfg, splits, force)


def list_runs() -> None:
    files = sorted(RUNS.glob("*.json"))
    print(f"{len(files)} runs in {RUNS}")
    for f in files:
        d = json.loads(f.read_text())
        c = d["config"]
        m = d["metrics"]
        key = (
            f"minR={m.get('minority_recall', float('nan')):.3f}"
            if "minority_recall" in m
            else f"auc={m.get('macro_auc', float('nan')):.3f}"
        )
        print(
            f"  {d['run_id']} {c['dataset']:14s} {c['method']:24s} "
            f"eps={str(c['eps']):5s} s={c['seed']} imb={str(c.get('imbalance_ratio')):5s} {key}"
        )


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", default="main", choices=["main", "imbalance", "chest", "all"])
    p.add_argument("--force", action="store_true")
    p.add_argument("--seeds", default=None, help="comma separated, e.g. 0,1")
    p.add_argument("--methods", default=None)
    p.add_argument("--eps", default=None, help="comma separated, 'none' for non-private")
    p.add_argument("--list", action="store_true")
    p.add_argument("--single", default=None, help="internal: JSON config for one isolated run")
    a = p.parse_args(argv)

    if a.single:
        payload = json.loads(a.single)
        cfg = TrainConfig(**payload)
        splits = _load_splits_for(cfg)
        _execute(cfg, splits)
        return 0

    if a.list:
        list_runs()
        return 0

    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else None
    methods = a.methods.split(",") if a.methods else None
    eps = None
    if a.eps:
        eps = [None if s.lower() in ("none", "null", "") else float(s) for s in a.eps.split(",")]

    if a.grid in ("main", "all"):
        grid_main(a.force, seeds, methods, eps)
    if a.grid in ("imbalance", "all"):
        grid_imbalance(a.force)
    if a.grid in ("chest", "all"):
        grid_chest(a.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
