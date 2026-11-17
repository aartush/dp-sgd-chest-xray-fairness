# Project Proposal — One Page

**Course:** Machine Learning Project (Ethics / Responsible AI track)
**Tushar Sharma** · **MTech I Year, NSUT** · **tushar.sharma.pg26@nsut.ac.in**
**Date:** 17 November 2026

---

### 1. Problem statement

Differential privacy (DP) is the standard guarantee for training models on sensitive medical data, and DP-SGD is its most-used mechanism. The open question this project addresses is **who pays the cost of that guarantee**. Under DP-SGD, every per-example gradient is clipped to a fixed norm and the average is noised. Clipping bounds the influence a single patient can have on the model, which is exactly what protects privacy — but it also erases the gradient signal from *rare* examples, because their unclipped updates are the ones most likely to exceed the bound. In an imbalanced clinical dataset the rare group is usually a clinically important minority (here, the *normal* class). The research question is therefore:

> **Does DP-SGD degrade recall for minority classes disproportionately, and can cheap, privacy-neutral interventions — a class-weighted loss or a pretrained initialisation — recover that loss without weakening the privacy guarantee?**

I will measure per-class recall and empirical privacy leakage (a membership-inference attack) across privacy budgets ε = {1, 3, 8} and no-DP, so that the utility–fairness–privacy trade-off can be read off a single table.

### 2. Selected papers and research gaps

1. **Huang & Xie, "Steps Adaptive Decay DPSGD: Enhancing Performance on Imbalanced Datasets with Differential Privacy with HAM10000," arXiv:2507.06619 (July 2025).** The paper states that gradient-noise DP methods "work well on large datasets like MNIST and CIFAR-100, but fail on small, imbalanced medical datasets like HAM10000," attributing this to minority-class gradients being clipped while majority classes dominate the update. Their remedy changes the *training schedule* (linearly decaying noise and clipping thresholds) and reports +2.15% accuracy at ε = 3.
   **Gap:** the paper reports only aggregate accuracy, not per-class recall, so it cannot show whether the minority class was actually fixed or simply swamped. It also does not test any architectural or loss-level intervention, and it is a two-author arXiv preprint with no peer review and no attack-based privacy evaluation.

2. **Mohammadi et al., "Differential privacy for medical deep learning: methods, tradeoffs, and deployment implications," arXiv:2506.00660 (May 2025); published in *npj Digital Medicine* 9:93 (2026).** A scoping review of 74 studies concluding that "severe degradation often occurs under strict privacy (ε ≈ 1), particularly in underrepresented or complex modalities," that "privacy-induced performance gaps disproportionately affect demographic subgroups," and that only a small subset of studies evaluates these tradeoffs at all. It names "balancing privacy with utility and fairness" as a critical open challenge.
   **Gap:** a review can identify the tension but cannot test a fix. No study in it isolates minority-class recall as the outcome, and none tests whether a trivial class-weighting change closes the gap.

**This project's contribution:** hold the DP mechanism fixed and vary only the loss (class weights) and the initialisation (ImageNet transfer), reporting **per-class recall** and a **membership-inference attack** rather than aggregate accuracy. Class weighting and pretrained init use only public class frequencies and public ImageNet weights, so neither consumes privacy budget — any recovery they produce is free under the same ε.

### 3. Dataset

**PneumoniaMNIST** (primary) — 5,856 paediatric chest X-rays, 28×28 grayscale, binary pneumonia-vs-normal, from the MedMNIST v2 benchmark (Yang et al., *Scientific Data* 10:41, 2023). Splits: 4,708 / 524 / 624 train/val/test. Genuinely imbalanced: train contains **1,214 normal vs 3,494 pneumonia** (26% / 74%). None of these are synthetically induced, so the imbalance reflects real acquisition.
**ChestMNIST** (scale check) — 78,468 frontal chest X-rays, 14 multi-label findings from NIH-ChestXray14; the rarest finding has only 144 positives in the 78,468-image training split. Used to test whether the findings transfer to a larger, multi-label, far more severe imbalance.
Both are public, de-identified, and require no data-use agreement; they download automatically through the `medmnist` package.

### 4. Software and tools

- **Python 3.12**, **PyTorch 2.6** (CUDA 12.4), **torchvision** — training and ImageNet initialisation.
- **Opacus 1.5.4** — DP-SGD with the RDP accountant; GroupNorm replaces BatchNorm so per-sample gradients are valid.
- **scikit-learn** — per-class metrics and the membership-inference classifier.
- **Hardware:** NVIDIA RTX 4050 Laptop GPU (6 GB). Batch size 32 is required for the DP arms because Opacus materialises per-sample gradients; this constraint and the resulting subdivision of privacy budget are reported with every run.
- **Code and artifacts:** all experiments run from a versioned repository; each run writes a JSON artifact recording seed, achieved ε, noise multiplier, clipping norm, step count and full history, so every reported number is traceable.

**Existing implementations explored.** Meta's **Opacus** (`github.com/meta-pytorch/opacus`, ~2k stars) supplies per-sample clipping, ghost clipping and the RDP accountant used here; the **MedMNIST** repo (`github.com/MedMNIST/MedMNIST`, ~1.4k stars) supplies the official loaders and splits. For the attack I reviewed **ML Privacy Meter** (`github.com/privacytrustlab/ml_privacy_meter`) and a PyTorch shadow-model implementation of Shokri et al. (`github.com/yonsei-sslab/MIA`). No official code exists for the SAD-DPSGD paper, and no established repository applies DP-SGD to chest X-rays on MedMNIST — only small unofficial attempts exist (e.g. DP fine-tuning on ChestX-ray14). The pipeline here is therefore built directly on Opacus primitives rather than a fork, which keeps the DP mechanics auditable.

### 5. Deliverables

Trained models at ε = {1, 3, 8} and no-DP across four methods (vanilla, class-weighted, pretrained, both); per-class recall and recall-gap tables; membership-inference AUC showing the empirical privacy gain; an imbalance-ratio sweep; and a 1-page results summary. All runs complete on the available GPU in a few hours total.
