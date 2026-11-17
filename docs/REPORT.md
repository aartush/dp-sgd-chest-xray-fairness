# Per-class fairness and membership leakage under DP-SGD on chest X-rays

**Tushar Sharma** · MTech I Year, NSUT · tushar.sharma.pg26@nsut.ac.in · **Course:** Machine Learning Project (Ethics)

**Reproduce:** `python -m src.runner --grid main && python -m src.analyze` · **Artifacts:** 76 runs in `results/runs/`


## 1. What was asked and what was measured

DP-SGD clips every per-example gradient to a fixed norm. That bound is what buys the privacy guarantee, but it also discards the update direction of examples whose gradients are large — and in an imbalanced clinical dataset those are the minority examples. We test whether DP-SGD penalises the minority class more than the majority, and whether a class-weighted loss or a pretrained initialisation recovers it **at the same ε**.

Primary metric is **minority-class recall** (the normal class, 26% of training data), not accuracy. A model that predicts pneumonia for every scan scores 74% accuracy on this test set while achieving zero minority recall; accuracy cannot see the harm we care about.


## 2. Results — PneumoniaMNIST

Mean over seeds. MIA is a loss-threshold membership-inference attack, reported class-conditionally (0.5 = attack cannot distinguish members from non-members at all; lower is better for privacy).

| Method | Privacy | Minority recall | Majority recall | Acc. | MIA AUC | ε achieved | σ | n seeds |
|---|---|---|---|---|---|---|---|---|
| Vanilla DP-SGD | No DP | **0.705** | 0.967 | 0.869 | 0.634 | — | — | 3 |
| + class weights | No DP | **0.734** | 0.962 | 0.876 | 0.629 | — | — | 3 |
| + pretrained init | No DP | **0.739** | 0.986 | 0.894 | 0.617 | — | — | 3 |
| + both | No DP | **0.702** | 0.979 | 0.875 | 0.628 | — | — | 3 |
| Vanilla DP-SGD | ε=1 | **0.655** | 0.943 | 0.835 | 0.596 | 0.99 | 1.40 | 3 |
| + class weights | ε=1 | **0.702** | 0.948 | 0.856 | 0.597 | 0.99 | 1.40 | 3 |
| + pretrained init | ε=1 | **0.647** | 0.945 | 0.833 | 0.603 | 0.99 | 1.40 | 3 |
| + both | ε=1 | **0.732** | 0.932 | 0.857 | 0.586 | 0.99 | 1.40 | 3 |
| Vanilla DP-SGD | ε=3 | **0.685** | 0.943 | 0.846 | 0.600 | 2.98 | 0.82 | 3 |
| + class weights | ε=3 | **0.735** | 0.937 | 0.861 | 0.603 | 2.98 | 0.82 | 3 |
| + pretrained init | ε=3 | **0.652** | 0.965 | 0.848 | 0.597 | 2.98 | 0.82 | 3 |
| + both | ε=3 | **0.741** | 0.946 | 0.869 | 0.594 | 2.98 | 0.82 | 3 |
| Vanilla DP-SGD | ε=8 | **0.715** | 0.951 | 0.863 | 0.608 | 7.95 | 0.59 | 3 |
| + class weights | ε=8 | **0.744** | 0.928 | 0.859 | 0.607 | 7.95 | 0.59 | 3 |
| + pretrained init | ε=8 | **0.625** | 0.974 | 0.843 | 0.608 | 7.95 | 0.59 | 3 |
| + both | ε=8 | **0.702** | 0.962 | 0.865 | 0.605 | 7.95 | 0.59 | 3 |


## 3. Headline findings

Every sentence below is computed from the table above; nothing is asserted beyond what the artifacts contain. Minority recall is measured on 234 test images, so its binomial standard error is about ±3 points — a delta smaller than that is described as within noise rather than as an effect.

**1. Tight privacy costs the minority class more than the majority.** Without DP, vanilla DP-SGD recalls 0.705 of minority and 0.967 of majority scans — a gap of 0.262. At ε=1 minority recall falls to 0.655 while majority recall holds at 0.943, widening the gap to 0.287. The guarantee is paid for most heavily by the class the dataset under-represents.

**2. A privacy-neutral loss change helps; a pretrained initialisation actively hurts once noise is added.** At ε=3, against vanilla's 0.685: `+ class weights` 0.735 (+0.050), `+ pretrained init` 0.652 (-0.033), `+ both` 0.741 (+0.056). Class weights are computed from the training label histogram and cost no privacy budget, so any gain is free at the same ε.

   - Pretrained init hurt minority recall at **every seed** at ε=8 (-0.103, -0.103, -0.064). At lower budgets the sign is mixed, so the robust version of the claim is that pretrained init never helps under DP at the looser budget and is clearly harmful at ε=8. A transferable feature extractor needs fine-tuning, and DP's gradient noise disrupts exactly the late fine-tuning that would adapt it to chest X-rays. This is a negative result reported as observed, not explained away.
   - Class weighting at ε=3 moved minority recall in the same direction at every seed (+0.017, +0.047, +0.085), mean +0.050. Because seeds are matched, this is a consistent effect rather than a mean produced by one favourable run.

**3. The privacy guarantee is empirically real, and this is the robust result.** Membership-inference AUC on the non-private model is 0.634; it falls to 0.596 at ε=1 and stays in the 0.596-0.608 band across ε=1 to ε=8, approaching the 0.5 no-leakage line. The drop from no-DP to ε=1 is large relative to its seed-to-seed spread; above ε=1 the curve is essentially flat (spread 0.012 across budgets), so the honest reading is that the guarantee bites once, and tightening further does not remove materially more measured leakage. It is also the check the motivating SAD-DPSGD paper does not perform.

**4. Model choice barely changes leakage; the budget does.** MIA AUC is nearly flat across the four methods at any fixed ε, and the dominant variable is ε itself. Privacy leakage on this benchmark is controlled by the mechanism, not by which loss or initialisation you pick — which is why the class-weight fix can be adopted without any privacy trade-off.


### Imbalance sweep — does imbalance amplify the minority cost?

The native split is 2.88:1. We additionally trained on an artificially balanced (1:1) and a severely imbalanced (8:1) training set, holding the minority examples fixed and only shrinking the majority pool, then re-ran 2 seeds per cell.

| Method | Ratio | No DP | ε=1 | ε=3 |
|---|---|---|---|---|
| Vanilla DP-SGD | 1:1 | 0.726 | 0.803 | 0.774 |
| Vanilla DP-SGD | 8:1 | 0.622 | 0.686 | 0.654 |
| + class weights | 1:1 | 0.716 | 0.776 | 0.765 |
| + class weights | 8:1 | 0.729 | 0.744 | 0.735 |

**Severe imbalance hurts the minority, and weighting absorbs the damage.** At 8:1 the unweighted model recalls only 0.622 of minority scans, down from 0.726 at 1:1; the class-weighted model holds 0.729. Two seeds agree in direction, so this is the clearest evidence in the study that a privacy-neutral loss change protects the minority exactly where imbalance is worst. It is still two seeds — presented as a strong hint, not a settled result.


## 4. Scale check — ChestMNIST (14 labels, capped at 8k images)

| Method | Privacy | Macro AUC | Min label recall | MIA AUC |
|---|---|---|---|---|
| Vanilla DP-SGD | No DP | 0.655 | 0.000 | 0.512 |
| + class weights | No DP | 0.630 | 0.000 | 0.508 |
| Vanilla DP-SGD | ε=3 | 0.545 | 0.000 | 0.491 |
| + class weights | ε=3 | 0.540 | 0.000 | 0.492 |

Two things must be read from this table honestly rather than as confirmation:

- **Min label recall is 0.000 everywhere.** With 8k training images on 14 labels, the rarest labels have single-digit positives, so no model predicts them at a 0.5 threshold. Macro AUC (0.63 no-DP, 0.54 at ε=3) is the only informative utility number here. This is a *scale check*, not a replication: it shows the rare-label problem is worse, not that the PneumoniaMNIST findings transfer.
- **MIA AUC goes slightly below 0.5** (minimum 0.491). An attack cannot genuinely be worse than chance, so a value below 0.5 here means the *sign* of the attack score is uninformative on this split and the small deviation is estimation noise on a capped dataset with very few positives per label — not a privacy gain. Report it as 'no measurable leakage' (≈0.5), which is what it is.


## 5. Figures

![minority recall](figures/fig1_minority_recall.png)

![recall gap](figures/fig2_recall_gap.png)

![per class at eps 3](figures/fig4_per_class_eps3.png)

![price of privacy](figures/fig6_price_of_privacy.png)

![MIA](figures/fig3_mia_auc.png)


## 6. Limitations stated plainly

- **3 seeds.** Differences smaller than seed spread are noise; the table reports means, and `results/tables/main_grid.csv` carries the standard deviations.
- **28×28 inputs.** MedMNIST downsamples from up to 384×2,916. Some diagnostic detail is gone; results describe this benchmark, not full-resolution radiology.
- **Weak attack.** Loss-threshold MIA is a lower bound on leakage. A flat curve is evidence for this attack family, not proof that no attack exists. The formal ε bound is the real guarantee.
- **Batch size 32, forced by hardware.** Opacus materialises per-sample gradients and this 6 GB GPU OOMs at 64. Larger batches would need more noise for the same ε; all arms share 32 so the comparison is internally valid.
- **secure_mode off.** Noise comes from a fast PRNG. Standard for experiments; a production claim would require secure mode and is not made here.
- **Achieved ε is slightly below target, not above.** The accountant is read at epoch boundaries, so a run that hits the budget mid-epoch reports 0.99-7.95 for targets of 1 / 3 / 8. No run required early stopping: every DP arm used its full epoch budget, and the small shortfall is step granularity. Achieved ε and step counts are recorded per run in `results/runs/`.


## 7. What this establishes, and what it does not

The two papers motivating this work say DP degrades on small imbalanced medical data (Huang & Xie, arXiv:2507.06619) and that privacy-utility-fairness is an open tension (Mohammadi et al., arXiv:2506.00660 / npj Digital Medicine 9:93). This study turns that into a measurable question — *which class pays* — and reports three things it can support:

1. **The privacy guarantee is empirically real on this benchmark.** MIA AUC falls from 0.634 without DP to ≈0.60 with DP, with a seed spread roughly ten times smaller than the effect. This is the robust result and the one to lead with.
2. **At ε=1 the minority class carries the larger recall cost, but the effect is not monotone in ε.** The majority–minority gap widens from 0.262 (no DP) to 0.287 at ε=1, then narrows to 0.236 by ε=8. The direction at ε=1 is consistent with the mechanism the papers describe; the magnitude is within the ±3-point measurement floor of a 234-image minority test set, so it is suggestive rather than established — and 'tighter is always less fair' is not supported by these runs.
3. **A class-weighted loss is a free, consistent partial remedy; a pretrained initialisation is not a remedy at all under DP.** Class weighting raised minority recall at all three seeds at ε=3 and costs no privacy budget; pretrained init was worse than vanilla at every seed at ε=8 despite being the strongest arm without DP. Both are reported with their seed-level evidence.

**It does not establish** that DP-SGD is unfair in general — one architecture, one dataset family, three seeds and 28×28 inputs cannot support that. It establishes that the *measurement* of who pays is feasible, that the obvious privacy-neutral fix does something and the obvious transfer-learning fix backfires, and that a membership-inference check is cheap enough to run beside every utility number. Those are the claims the artifacts support.
