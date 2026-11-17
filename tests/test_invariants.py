"""Invariant tests for the parts that would silently corrupt results if broken.

These are deliberately not "does it run" tests. Each one pins a property that,
if violated, would make a reported number wrong without raising an error:

  * class weights must favour the minority class
  * imbalance construction must leave the minority set untouched
  * MIA AUC must be ~0.5 on random scores and high on separable ones
  * the DP accountant must actually spend the budget it claims
  * the model must contain no BatchNorm (Opacus would reject it)
  * pretrained init must differ from random init and be reproducible

Run:  .venv/Scripts/python.exe -m pytest tests -q
   or .venv/Scripts/python.exe tests/test_invariants.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import Split, class_balanced_weights, make_imbalanced_train, multilabel_pos_weights
from src.metrics import binary_report
from src.mia import loss_threshold_attack, per_example_loss, tpr_at_fpr
from src.models import build_model, count_parameters


# ---------------------------------------------------------------- data / weights


def test_class_weights_favor_minority():
    labels = np.array([0] * 10 + [1] * 90)
    w = class_balanced_weights(labels, 2)
    assert w[0] > w[1], "minority class must receive the larger weight"
    assert np.isclose(w.mean(), 1.0), "weights are normalised to mean 1"
    # inverse frequency: 90/10 ratio
    assert np.isclose(w[0] / w[1], 9.0, rtol=1e-6)


def test_multilabel_pos_weights_are_inverse_frequency():
    labels = np.zeros((100, 2), dtype=np.int64)
    labels[:10, 0] = 1  # rare label
    labels[:50, 1] = 1  # common label
    w = multilabel_pos_weights(labels)
    assert w[0] > w[1]
    assert np.isclose(w.mean(), 1.0)


def test_imbalance_keeps_minority_intact():
    rng = np.random.default_rng(0)
    labels = np.array([0] * 100 + [1] * 1000)
    images = rng.normal(size=(len(labels), 1, 4, 4)).astype(np.float32)
    split = Split(images, labels, "train")
    out = make_imbalanced_train(split, ratio=5.0, minority_class=0, seed=0)
    assert (out.labels == 0).sum() == 100, "minority examples must all survive"
    assert (out.labels == 1).sum() == 500, "majority downsampled to ratio x minority"
    # the surviving minority rows are byte-identical to the originals
    original_minority = images[labels == 0]
    kept_minority = out.images[out.labels == 0]
    assert np.array_equal(original_minority, kept_minority)


def test_imbalance_noop_when_ratio_already_met():
    labels = np.array([0] * 10 + [1] * 10)
    split = Split(np.zeros((20, 1, 4, 4), dtype=np.float32), labels, "train")
    out = make_imbalanced_train(split, ratio=5.0, minority_class=0)
    assert len(out) == 20


# ---------------------------------------------------------------- metrics


def test_minority_recall_is_about_the_minority_class():
    # model predicts majority (1) for everything: minority recall must be 0
    y = np.array([0, 0, 1, 1, 1])
    pred = np.array([1, 1, 1, 1, 1])
    rep = binary_report(y, pred.astype(float), threshold=0.5)
    assert rep["recall_class0"] == 0.0
    assert rep["recall_class1"] == 1.0
    assert rep["minority_recall"] == 0.0
    assert rep["gap_majority_minus_minority"] == 1.0


def test_majority_class_collapse_scores_high_accuracy_but_zero_minority_recall():
    """The exact failure mode the project exists to expose."""
    y = np.array([0] * 26 + [1] * 74)
    probs = np.ones(len(y))  # always pneumonia
    rep = binary_report(y, probs)
    assert rep["accuracy"] == pytest.approx(0.74)
    assert rep["minority_recall"] == 0.0


# ---------------------------------------------------------------- MIA


def test_mia_auc_is_chance_for_indistinguishable_scores():
    rng = np.random.default_rng(0)
    tr = rng.normal(1.0, 1.0, 500)
    te = rng.normal(1.0, 1.0, 500)
    lab = np.zeros(500, dtype=int)
    out = loss_threshold_attack(tr, lab, te, lab)
    assert 0.42 < out["mia_auc_pooled"] < 0.58, "uninformative losses must give ~0.5 AUC"


def test_mia_detects_memorisation():
    # members have much lower loss => high AUC
    tr = np.full(500, 0.01)
    te = np.full(500, 2.0)
    lab = np.array([0, 1] * 250)
    out = loss_threshold_attack(tr, lab, te, lab)
    assert out["mia_auc_pooled"] > 0.95


def test_tpr_at_fpr_extremes():
    members = np.array([10.0, 9.0, 8.0])
    nonmembers = np.array([0.0, 0.0, 0.0])
    assert tpr_at_fpr(members, nonmembers, 0.10) == pytest.approx(1.0)
    assert tpr_at_fpr(nonmembers, members, 0.10) == pytest.approx(0.0)


def test_per_example_loss_matches_manual_cross_entropy():
    logits = np.array([[2.0, 0.0], [0.0, 2.0]])
    labels = np.array([0, 1])
    loss = per_example_loss(logits, labels)
    # both examples are correct and confident; loss should be small and equal
    assert loss[0] == pytest.approx(loss[1])
    assert loss[0] < 0.2


# ---------------------------------------------------------------- shadow attack


def _draw(rng, n, leak=0.0):
    """Members (leak>0) get a confidence boost on their true class."""
    lab = rng.integers(0, 2, n)
    logits = rng.normal(0, 1.0, (n, 2))
    if leak:
        logits[np.arange(n), lab] += leak
    return logits, lab


def test_shadow_attack_is_at_chance_without_leakage():
    """The null: members and non-members drawn from the same distribution.

    A shadow attack that reports leakage here would be a false positive, which
    is the failure mode that would let this project claim a privacy effect that
    does not exist.
    """
    from src.mia import shadow_model_attack

    rng = np.random.default_rng(0)
    n = 800
    si, sl = _draw(rng, n)
    so, slo = _draw(rng, n)
    ti, tl = _draw(rng, n)
    to, tlo = _draw(rng, n)
    out = shadow_model_attack(ti, tl, to, tlo, si, sl, so, slo)
    assert 0.42 < out["mia_shadow_transfer_auc"] < 0.58, (
        f"no-leakage case gave AUC {out['mia_shadow_transfer_auc']:.3f}; the attack is biased"
    )


def test_shadow_attack_detects_injected_leakage():
    from src.mia import shadow_model_attack

    rng = np.random.default_rng(1)
    n = 800
    si, sl = _draw(rng, n, leak=1.5)
    so, slo = _draw(rng, n, leak=0.0)
    ti, tl = _draw(rng, n, leak=1.5)
    to, tlo = _draw(rng, n, leak=0.0)
    out = shadow_model_attack(ti, tl, to, tlo, si, sl, so, slo)
    assert out["mia_shadow_transfer_auc"] > 0.6, (
        f"injected leakage was not detected (AUC {out['mia_shadow_transfer_auc']:.3f})"
    )


# ---------------------------------------------------------------- model


def test_model_has_no_batchnorm():
    m = build_model("resnet18", 2)
    bn = [n for n, mod in m.named_modules() if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)]
    assert bn == [], f"Opacus rejects BatchNorm; found {bn}"


def test_model_has_no_inplace_relu():
    m = build_model("resnet18", 2)
    bad = [
        n
        for n, mod in m.named_modules()
        if isinstance(mod, (torch.nn.ReLU, torch.nn.ReLU6)) and mod.inplace
    ]
    assert bad == [], f"in-place ReLU breaks Opacus hooks: {bad}"


def test_pretrained_differs_from_scratch_and_is_deterministic():
    a = build_model("resnet18", 2, pretrained=False)
    b = build_model("resnet18", 2, pretrained=True)
    c = build_model("resnet18", 2, pretrained=True)
    wa = a.layer1[0].conv1.weight
    wb = b.layer1[0].conv1.weight
    assert not torch.allclose(wa, wb), "pretrained weights must actually transfer"
    assert torch.allclose(wb, c.layer1[0].conv1.weight), "transfer must be deterministic"


def test_forward_shape():
    m = build_model("resnet18", 2).eval()
    with torch.no_grad():
        out = m(torch.randn(3, 1, 28, 28))
    assert out.shape == (3, 2)


# ---------------------------------------------------------------- DP accounting


def test_dp_epsilon_increases_with_steps():
    """A quieter check than a full training run: the accountant must behave."""
    from opacus.accountants import RDPAccountant

    acc = RDPAccountant()
    eps_at = []
    for step in range(1, 200):
        acc.step(noise_multiplier=0.8, sample_rate=0.02)
        if step in (50, 100, 199):
            eps_at.append(acc.get_epsilon(delta=1e-5))
    assert eps_at[0] < eps_at[1] < eps_at[2], "epsilon must grow with more steps"
    assert all(e > 0 for e in eps_at)


def test_noise_multiplier_decreases_with_larger_epsilon():
    from opacus.accountants.utils import get_noise_multiplier

    nm1 = get_noise_multiplier(target_epsilon=1.0, target_delta=1e-5, sample_rate=0.01, steps=1000, accountant="rdp")
    nm8 = get_noise_multiplier(target_epsilon=8.0, target_delta=1e-5, sample_rate=0.01, steps=1000, accountant="rdp")
    assert nm1 > nm8, "a stricter privacy budget must require more noise"


def test_privacy_engine_rejects_batchnorm():
    """Negative control: confirms the GroupNorm choice is load-bearing."""
    from opacus import PrivacyEngine
    from opacus.validators.errors import UnsupportedModuleError
    import torch.nn as nn

    net = nn.Sequential(nn.Conv2d(1, 4, 3), nn.BatchNorm2d(4))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(8, 1, 8, 8), torch.zeros(8, dtype=torch.long)),
        batch_size=4,
    )
    with pytest.raises(UnsupportedModuleError):
        PrivacyEngine(secure_mode=False).make_private(
            module=net,
            optimizer=torch.optim.SGD(net.parameters(), lr=0.1),
            data_loader=loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )


def test_param_count_is_sane():
    # ResNet-18 is ~11.2M for 2 classes; a 1-channel stem is slightly smaller
    n = count_parameters(build_model("resnet18", 2))
    assert 10_000_000 < n < 12_000_000, f"unexpected parameter count {n}"


# ---------------------------------------------------------------- loss shapes


def test_multilabel_weighted_loss_shapes_and_effect():
    """Regression: the multilabel weighted loss once broadcast (B,) against (B,L).

    It must keep the (B, L) shape, reduce to one value per example, and actually
    give a rare-label positive more weight than a common-label positive.
    """
    import torch as _t
    from src.train import _per_example_loss, WeightedLoss

    logits = _t.zeros(4, 3)
    labels = _t.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    w = _t.tensor([10.0, 1.0, 1.0])

    per_ex = _per_example_loss(logits, labels, class_weight=w, multilabel=True)
    assert per_ex.shape == (4,), f"expected one loss per example, got {per_ex.shape}"

    # a confident wrong prediction on the rare label must cost more
    wrong_rare = _t.tensor([[0.0, 4.0, 4.0], [4.0, 0.0, 4.0], [4.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    rare_vs_common = _per_example_loss(wrong_rare, labels, class_weight=w, multilabel=True)
    assert rare_vs_common[0] > rare_vs_common[1], "rare-label error must be weighted higher"

    crit = WeightedLoss(w, multilabel=True)
    scalar = crit(logits, labels)
    assert scalar.dim() == 0 and _t.isfinite(scalar)


def test_multilabel_weighted_loss_without_weights_matches_plain_bce():
    import torch as _t
    from src.train import _per_example_loss

    logits = _t.randn(5, 3)
    labels = _t.randint(0, 2, (5, 3))
    got = _per_example_loss(logits, labels, class_weight=None, multilabel=True)
    import torch.nn.functional as F

    want = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none").mean(dim=1)
    assert _t.allclose(got, want)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
