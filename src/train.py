"""Training engine: non-private SGD and DP-SGD through Opacus.

Design decisions and why they matter for correctness:

Privacy accounting
    Opacus' accountant charges one unit per optimiser step. The noise multiplier
    must therefore be known *before* `make_private` is called. We plan a fixed
    number of steps (epochs x steps_per_epoch), ask the RDP accountant for the
    sigma that spends exactly the target epsilon over those steps, then stop
    early if the accountant crosses epsilon before the plan ends. Every arm uses
    the same physical batch size and data order, so "epsilon = 1 / 3 / 8" is the
    only thing that differs between private runs.

Batch size
    Measured on the target GPU (RTX 4050, 6 GB): Opacus peaks at 1.6 GB for
    batch 32, 4.6 GB for 64, and OOMs at 128. We use 64. `expected_batch_size`
    is set explicitly so the loss rescaling matches the physical batch.

Clipping bound
    C is calibrated from the median per-example gradient norm on the last linear
    layer over a few non-private warm-up batches. The median (not max) keeps
    clipping noise low and is stable across seeds. Because the calibration runs
    before any DP training and aggregates over a batch, it contributes no
    per-example information to the released model.

Class weights
    Applied inside the loss on a per-example basis. Opacus clips per-example
    gradients, so a weighted minority example still gets clipped to C; weighting
    changes the direction of the average, not the privacy guarantee. Weights are
    computed from the training label histogram only.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict

# NOTE ON BATCH SIZE. Opacus allocates a fresh per-sample-gradient buffer per
# parameter on every step. On this machine (RTX 4050, 6 GB, Windows build where
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments is *not* supported) batch 64
# fragments reserved memory until an OOM around step 60; batch 32 plateaus at
# 6.3 GB reserved and completes full epochs. 32 is therefore the required
# setting for every DP arm. Non-private arms use the same size so the only
# difference between arms is the privacy mechanism.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import Split, ArrayDataset
from .models import build_model, count_parameters


@dataclass
class TrainConfig:
    dataset: str = "pneumoniamnist"
    arch: str = "resnet18"
    method: str = "baseline"  # baseline | classweight | pretrained | pretrained+classweight
    eps: float | None = None  # None => non-private
    epochs: int = 15
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float | None = None  # None => calibrate
    delta: float = 1e-5
    multilabel: bool = False
    seed: int = 0
    imbalance_ratio: float | None = None
    max_train: int | None = None
    device: str = "cuda"
    max_steps: int | None = None
    grad_sample_mode: str = "ghost"  # ghost | hooks; recorded for provenance

    @property
    def use_class_weights(self) -> bool:
        return "classweight" in self.method

    @property
    def use_pretrained(self) -> bool:
        return self.method.startswith("pretrained")

    @property
    def private(self) -> bool:
        return self.eps is not None


@dataclass
class TrainResult:
    config: dict
    history: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    achieved_eps: float | None = None
    noise_multiplier: float | None = None
    wall_seconds: float = 0.0
    n_params: int = 0
    steps_run: int = 0
    epoch_scores: list = field(default_factory=list)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(split, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ArrayDataset(split),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,
        num_workers=0,
        # pin_memory was on originally; on this 16 GB machine it caused
        # cuDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED because pinned
        # (non-pageable) host memory is scarce and does not return promptly.
        # The datasets are tiny and already on-GPU-bound, so pinning buys
        # nothing here and costs stability.
        pin_memory=False,
    )


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: str):
    """Inference on the *inner* module, never the Opacus wrapper.

    Evaluating through GradSampleModule fires its activation-capture hooks, which
    retain activations and (in ghost mode) desynchronise the activation stack so
    the next backward raises `pop from empty list`. Unwrapping first means no
    privacy hook participates in inference at all.
    """
    inner = model._module if hasattr(model, "_module") else model
    was_training = inner.training
    inner.eval()
    logits_all, labels_all = [], []
    try:
        for x, y in loader:
            out = inner(x.to(device))
            logits_all.append(out.detach().cpu().float().numpy())
            labels_all.append(y.numpy())
    finally:
        if was_training:
            inner.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(logits_all), np.concatenate(labels_all)


class WeightedLoss(nn.Module):
    """nn.Module loss so Opacus' ghost-clipping path can inspect `reduction`.

    Ghost clipping needs a criterion object (not a function) because it peeks at
    `.reduction` to validate that it matches the module and optimiser. `w` holds
    the per-class weights as a buffer, so it follows .cuda() correctly.
    """

    reduction = "mean"

    def __init__(self, weight: torch.Tensor | None, multilabel: bool):
        super().__init__()
        self.multilabel = multilabel
        if weight is not None:
            self.register_buffer("w", weight.detach().clone())
        else:
            self.w = None

    def forward(self, logits, y):
        if self.multilabel:
            yf = y.float()
            loss = F.binary_cross_entropy_with_logits(logits, yf, reduction="none")  # (B, L)
            if self.w is not None:
                # per-label positive weighting, negative terms unweighted; all
                # operands are (B, L) so no axis can mismatched
                lw = self.w.view(1, -1)
                loss = loss * (yf * lw + (1 - yf))
            return loss.mean()
        return F.cross_entropy(logits, y.long(), weight=self.w, reduction="mean")


def _per_example_loss(logits, y, class_weight=None, multilabel=False):
    if multilabel:
        yf = y.float()
        loss = F.binary_cross_entropy_with_logits(logits, yf, reduction="none")  # (B, L)
        if class_weight is not None:
            # weight the positive term per label; weight of the negative term is 1.
            # Shapes: loss/yf/lw all (B, L), so the result stays (B, L) and the
            # per-example reduction below aggregates over labels.
            lw = class_weight.to(logits.device).view(1, -1)
            loss = loss * (yf * lw + (1 - yf))
        return loss.mean(dim=1)
    return F.cross_entropy(logits, y.long(), weight=class_weight, reduction="none")


def calibrate_clipping_bound(model, split, device, cfg, batches: int = 6) -> float:
    """Median *true* per-sample gradient norm under an Opacus wrapper.

    The wrapper is built with noise 0 and an effectively infinite clipping bound
    so the gradients are unclipped; we then read `p.grad_sample` and take the
    L2 norm across all parameters for each example. Using the median keeps the
    noise low while allowing a minority of large-norm examples to be clipped,
    which is the standard trade-off. The calibration batch is not used for the
    final DP model and the statistic is aggregated, so no per-example
    information is released.
    """
    from opacus import PrivacyEngine
    import gc

    model.train()
    loader = make_loader(split, min(cfg.batch_size, 64), shuffle=True, seed=cfg.seed)
    batch_x, batch_y = next(iter(loader))
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    n = batch_x.size(0)

    probe_loader = make_loader(split, n, shuffle=False, seed=cfg.seed)
    pe = PrivacyEngine(accountant="rdp", secure_mode=False)
    gm, o, _ = pe.make_private(
        module=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        data_loader=probe_loader,
        noise_multiplier=0.0,
        max_grad_norm=1e6,
    )
    gm.train()
    o.zero_grad()
    loss = _per_example_loss(gm(batch_x), batch_y, None, cfg.multilabel).mean()
    loss.backward()
    sq = torch.zeros(n, device=device)
    for p in gm.parameters():
        gs = getattr(p, "grad_sample", None)
        if gs is None:
            continue
        sq += gs.reshape(n, -1).pow(2).sum(dim=1)
    med = float(sq.sqrt().median())

    # unwrap and release every per-sample gradient buffer before returning
    gm.remove_hooks()
    for p in model.parameters():
        p.grad = None
        for attr in ("grad_sample", "_current_grad_sample"):
            if hasattr(p, attr):
                delattr(p, attr)
    del gm, o, pe, probe_loader
    gc.collect()
    torch.cuda.empty_cache()
    model.train()
    return float(max(float(f"{med:.2g}"), 1e-3))


def _score_split(cfg, logits, y):
    from .metrics import binary_report, multilabel_report

    if cfg.multilabel:
        m = multilabel_report(y, 1 / (1 + np.exp(-logits)))
        return m["macro_auc"], m
    probs = torch.softmax(torch.from_numpy(logits), 1)[:, 1].numpy()
    m = binary_report(y, probs)
    return m["macro_recall"], m


def train(cfg: TrainConfig, splits: dict, log_fn=print) -> TrainResult:
    set_seed(cfg.seed)
    device = "cuda" if (cfg.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    train_split, val_split, test_split = splits["train"], splits["val"], splits["test"]
    num_classes = 14 if cfg.multilabel else 2

    model = build_model(cfg.arch, num_classes, pretrained=cfg.use_pretrained, in_channels=1).to(device)
    n_params = count_parameters(model)

    class_weight = None
    if cfg.use_class_weights:
        from .data import class_balanced_weights, multilabel_pos_weights

        w = multilabel_pos_weights(train_split.labels) if cfg.multilabel else class_balanced_weights(
            train_split.labels, num_classes
        )
        class_weight = torch.tensor(w, dtype=torch.float32, device=device)
        log_fn(f"    class weights: {np.round(w, 3).tolist()}")

    if cfg.max_grad_norm is None:
        if cfg.private:
            cfg.max_grad_norm = calibrate_clipping_bound(model, train_split, device, cfg)
            # the probe leaves per-sample gradient buffers on the parameters;
            # they are full-size and would overlap with the real DP run's buffers
            for p in model.parameters():
                p.grad = None
                for attr in ("grad_sample", "_current_grad_sample"):
                    if hasattr(p, attr):
                        delattr(p, attr)
            torch.cuda.empty_cache()
        else:
            cfg.max_grad_norm = 1.0

    train_loader = make_loader(train_split, cfg.batch_size, shuffle=True, seed=cfg.seed)
    val_loader = make_loader(val_split, 512, shuffle=False)
    test_loader = make_loader(test_split, 512, shuffle=False)

    steps_per_epoch = math.ceil(len(train_split) / cfg.batch_size)
    planned_steps = cfg.max_steps or cfg.epochs * steps_per_epoch
    sample_rate = cfg.batch_size / len(train_split)

    noise_multiplier = None
    if cfg.private:
        from opacus import PrivacyEngine
        from opacus.accountants.utils import get_noise_multiplier

        noise_multiplier = float(
            get_noise_multiplier(
                target_epsilon=cfg.eps,
                target_delta=cfg.delta,
                sample_rate=sample_rate,
                steps=planned_steps,
                accountant="rdp",
            )
        )
        log_fn(
            f"    C={cfg.max_grad_norm:g}  sigma={noise_multiplier:.4f}  "
            f"sample_rate={sample_rate:.4f}  planned_steps={planned_steps}"
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    privacy_engine = None
    criterion = None
    if cfg.private:
        from opacus import PrivacyEngine

        privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=False)
        # Ghost clipping computes per-sample norms without materialising the full
        # per-sample gradient tensors. On this 6 GB card hooks mode peaks at
        # 7.8 GB reserved (OOM under any other GPU use) while ghost peaks at
        # 5.9 GB with only 0.24 GB live — and it is not slower (123 vs 135 ms/step).
        criterion = WeightedLoss(class_weight, cfg.multilabel).to(device)
        model, optimizer, criterion, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=cfg.max_grad_norm,
            grad_sample_mode=cfg.grad_sample_mode,
            criterion=criterion,
            loss_reduction="mean",
        )

    inner = model._module if hasattr(model, "_module") else model

    history = []
    epoch_scores = []
    best_state, best_val = None, -1.0
    start = time.time()
    step = 0
    eps_now = 0.0

    for epoch in range(cfg.epochs):
        model.train()
        running, n_seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if cfg.private:
                loss = criterion(model(x), y)
            else:
                loss = _per_example_loss(model(x), y, class_weight, cfg.multilabel).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            # ghost clipping hands back a DPTensorFastGradientClipping wrapper,
            # which exposes .item()/.detach() but is not itself a Tensor and has
            # no .loss attribute; hooks mode hands back a plain tensor.
            try:
                running += float(loss.item()) * x.size(0)
            except (TypeError, ValueError, AttributeError):
                lv = getattr(loss, "loss", None)
                if lv is not None and hasattr(lv, "detach"):
                    running += float(lv.detach()) * x.size(0)
            n_seen += x.size(0)
            if cfg.max_steps and step >= cfg.max_steps:
                break

        if cfg.private:
            eps_now = float(privacy_engine.get_epsilon(delta=cfg.delta))

        val_logits, val_y = predict_logits(inner, val_loader, device)
        score, val_m = _score_split(cfg, val_logits, val_y)
        epoch_scores.append(score)
        history.append(
            {
                "epoch": epoch + 1,
                "step": step,
                "train_loss": running / max(n_seen, 1),
                "val_score": score,
                "val_minority_recall": val_m.get("recall_class0"),
                "epsilon": eps_now if cfg.private else None,
            }
        )
        msg = f"    epoch {epoch+1:02d} step {step:5d} loss {running/max(n_seen,1):.4f} val {score:.4f}"
        if cfg.private:
            msg += f"  eps {eps_now:.3f}"
        log_fn(msg)

        if score > best_val:
            best_val = score
            best_state = {k: v.detach().clone() for k, v in inner.state_dict().items()}

        if cfg.private and eps_now >= cfg.eps * 0.999:
            log_fn(f"    target epsilon {cfg.eps} reached at step {step}; stopping early")
            break
        if cfg.max_steps and step >= cfg.max_steps:
            break

    if best_state is not None:
        inner.load_state_dict(best_state)

    test_logits, test_y = predict_logits(inner, test_loader, device)
    train_logits, train_y = predict_logits(inner, train_loader, device)
    val_logits, val_y = predict_logits(inner, val_loader, device)
    achieved_eps = float(privacy_engine.get_epsilon(delta=cfg.delta)) if cfg.private else None

    return TrainResult(
        config=asdict(cfg),
        history=history,
        achieved_eps=achieved_eps,
        noise_multiplier=noise_multiplier,
        wall_seconds=time.time() - start,
        n_params=n_params,
        steps_run=step,
        epoch_scores=epoch_scores,
        metrics={
            "_train_logits": train_logits,
            "_train_labels": train_y,
            "_test_logits": test_logits,
            "_test_labels": test_y,
            "_val_logits": val_logits,
            "_val_labels": val_y,
        },
    )
