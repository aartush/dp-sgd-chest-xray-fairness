"""Membership inference attack (loss-threshold / Yeom-style, plus a learned variant).

Given a trained target model we score every example by its negative
cross-entropy loss: low loss => more likely a training member. The attack's
ROC-AUC over (train = members, test = non-members) quantifies memorisation.

Class-composition confound (important): PneumoniaMNIST's train and test splits
have different class mixes (26% vs 37% normal). A pooled attack AUC would then
partly measure "which cohort is this" rather than memorisation. We therefore
report a *class-conditional* AUC (computed within each class and averaged) as
the primary number, and the pooled AUC only as a secondary diagnostic.

An AUC near 0.5 means the attack cannot distinguish members from non-members,
i.e. no measurable privacy leakage from this attack family. Lower AUC under DP
is the empirical privacy gain the assignment asks us to demonstrate.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


def per_example_loss(logits: np.ndarray, labels: np.ndarray, multilabel: bool = False) -> np.ndarray:
    if multilabel:
        p = 1.0 / (1.0 + np.exp(-logits))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean(axis=1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1))
    logp = shifted - logsumexp[:, None]
    return -logp[np.arange(len(labels)), labels.astype(int)]


def tpr_at_fpr(member_scores: np.ndarray, nonmember_scores: np.ndarray, fpr_target: float) -> float:
    """TPR of a score-based attack at a fixed false-positive rate."""
    neg = np.sort(nonmember_scores)[::-1]
    if len(neg) == 0:
        return float("nan")
    k = max(int(np.ceil(fpr_target * len(neg))) - 1, 0)
    thr = neg[k] if k < len(neg) else -np.inf
    return float((member_scores > thr).mean())


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def loss_threshold_attack(
    train_loss: np.ndarray,
    train_labels: np.ndarray,
    test_loss: np.ndarray,
    test_labels: np.ndarray,
    multilabel: bool = False,
) -> dict:
    """Score = -loss. Higher score => more likely member."""
    s_mem, s_non = -train_loss, -test_loss
    pooled_auc = _safe_auc(
        np.concatenate([np.ones(len(s_mem)), np.zeros(len(s_non))]),
        np.concatenate([s_mem, s_non]),
    )

    per_class_auc = {}
    if multilabel:
        # per-label presence: a sample is a "member" for label j if it carries j
        for j in range(train_labels.shape[1]):
            m = train_labels[:, j] == 1
            n = test_labels[:, j] == 1
            if m.sum() < 10 or n.sum() < 10:
                continue
            per_class_auc[f"label{j}"] = _safe_auc(
                np.concatenate([np.ones(m.sum()), np.zeros(n.sum())]),
                np.concatenate([s_mem[m], s_non[n]]),
            )
    else:
        for c in (0, 1):
            m = train_labels.astype(int) == c
            n = test_labels.astype(int) == c
            if m.sum() < 10 or n.sum() < 10:
                continue
            per_class_auc[f"class{c}"] = _safe_auc(
                np.concatenate([np.ones(m.sum()), np.zeros(n.sum())]),
                np.concatenate([s_mem[m], s_non[n]]),
            )

    vals = [v for v in per_class_auc.values() if not np.isnan(v)]
    conditional_auc = float(np.mean(vals)) if vals else float("nan")

    return {
        "mia_auc_pooled": pooled_auc,
        "mia_auc_class_conditional": conditional_auc,
        "mia_advantage": (2 * pooled_auc - 1) if not np.isnan(pooled_auc) else float("nan"),
        "mia_tpr_at_1pct_fpr": tpr_at_fpr(s_mem, s_non, 0.01),
        "mia_tpr_at_10pct_fpr": tpr_at_fpr(s_mem, s_non, 0.10),
        "mia_mean_member_loss": float(train_loss.mean()),
        "mia_mean_nonmember_loss": float(test_loss.mean()),
        **{f"mia_auc_{k}": v for k, v in per_class_auc.items()},
    }


def learned_attack(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    train_is_member: np.ndarray,
    eval_feats: np.ndarray,
    eval_is_member: np.ndarray,
) -> dict:
    """Attack-model variant: logistic regression on (loss, confidence, entropy).

    Fitted on a disjoint shadow split supplied by the caller so the reported AUC
    is on data the attack model never saw.
    """
    if len(np.unique(eval_is_member)) < 2:
        return {"mia_learned_auc": float("nan")}
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(train_feats, train_is_member)
    scores = clf.predict_proba(eval_feats)[:, 1]
    return {"mia_learned_auc": _safe_auc(eval_is_member, scores)}


def attack_features(logits: np.ndarray, labels: np.ndarray, multilabel: bool = False) -> np.ndarray:
    """Features an attacker can compute from model outputs alone."""
    loss = per_example_loss(logits, labels, multilabel)
    if multilabel:
        p = 1.0 / (1.0 + np.exp(-logits))
        conf = np.abs(p - 0.5).mean(axis=1)
        ent = -(p * np.log(np.clip(p, 1e-7, 1)) + (1 - p) * np.log(np.clip(1 - p, 1e-7, 1))).mean(axis=1)
    else:
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        conf = p.max(axis=1)
        ent = -(p * np.log(np.clip(p, 1e-9, 1))).sum(axis=1)
    return np.stack([loss, conf, ent], axis=1)


def shadow_model_attack(
    target_train_logits: np.ndarray,
    target_train_labels: np.ndarray,
    target_test_logits: np.ndarray,
    target_test_labels: np.ndarray,
    shadow_in_logits: np.ndarray,
    shadow_in_labels: np.ndarray,
    shadow_out_logits: np.ndarray,
    shadow_out_labels: np.ndarray,
    multilabel: bool = False,
    seed: int = 0,
) -> dict:
    """Learned attack trained on a *separate* model's in/out behaviour.

    A threshold on loss assumes the attacker knows the right statistic. This
    variant instead learns a classifier over (loss, confidence, entropy) from a
    shadow model — same architecture, same data distribution, same DP settings —
    and then applies that classifier to the target model.

    The asymmetry that makes it a fair test: the shadow attacker is trained on
    the shadow model's own members vs non-members, never on the target model's.
    If the target model leaks, the learned rule transfers and AUC stays above
    0.5; if DP has flattened the loss distribution, the transferred rule decays
    toward chance. Reporting both attacks makes the privacy claim rest on two
    independent procedures rather than one hand-built score.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    def feats(logits, labels):
        return attack_features(logits, labels, multilabel)

    X_shadow = np.vstack(
        [
            feats(shadow_in_logits, shadow_in_labels),
            feats(shadow_out_logits, shadow_out_labels),
        ]
    )
    y_shadow = np.concatenate(
        [np.ones(len(shadow_in_labels)), np.zeros(len(shadow_out_labels))]
    )

    # hold part of the shadow data back so the attacker's own score is honest
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_shadow, y_shadow, test_size=0.3, random_state=seed, stratify=y_shadow
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_fit, y_fit)
    shadow_holdout_auc = _safe_auc(y_val, clf.predict_proba(X_val)[:, 1])

    X_target = np.vstack(
        [
            feats(target_train_logits, target_train_labels),
            feats(target_test_logits, target_test_labels),
        ]
    )
    y_target = np.concatenate([np.ones(len(target_train_labels)), np.zeros(len(target_test_labels))])
    target_scores = clf.predict_proba(X_target)[:, 1]

    pooled = _safe_auc(y_target, target_scores)
    n_mem = len(target_train_labels)
    s_mem, s_non = target_scores[:n_mem], target_scores[n_mem:]

    return {
        "mia_shadow_holdout_auc": shadow_holdout_auc,
        "mia_shadow_transfer_auc": pooled,
        "mia_shadow_tpr_at_1pct_fpr": tpr_at_fpr(s_mem, s_non, 0.01),
        "mia_shadow_n_shadow": int(len(y_shadow)),
    }

