# SRNet-v11 detector-development and secondary robustness protocol

## 1. Motivation

Patch 10 completed SRNet fitting but failed the pre-test development gate:
approximately random discrimination was observed on the held-out development
partition (`AUC ≈ 0.508`), while TEST was explicitly not scored. The falling
training loss together with near-chance development AUC motivated a revised
**detector-only** training protocol.

The RDH allocator, payloads, teacher, TEST source IDs, and all frozen primary
results remain unchanged.

## 2. Relation to published SRNet training

The original SRNet work used Adamax, 16 cover-stego pairs per minibatch,
paired rotations/mirroring, BatchNorm moving-average decay 0.9, He
initialization, L2 regularization of 2e-4, and a substantially longer training
schedule. It also used validation-snapshot selection and curriculum training
for lower payloads.

Patch 11 adopts the same qualitative principles but remains explicitly
CPU-constrained. It therefore reports an **SRNet-architecture detector** and
does not claim a reproduction of the original training regime.

## 3. Frozen RDH quantities

No modification is permitted to

\[
\alpha=0.25,\qquad R^*_{net}=0.009\ \mathrm{bpp},
\]

the SRM-derived teacher, the joint/predictability allocation rules, payload
accounting, frozen test source IDs, or notebook-06 results.

## 4. Detector-development data

Only the 6,000-image TRAIN split is used before the gate:

- fitting pool: TRAIN indices 0--5249 (5,250 sources);
- development: TRAIN indices 5250--5999 (750 sources).

Patch 11 no longer requires fitting-source disjointness from the enhanced CNN.
The intended independence is architectural/model independence: the teacher is
SRM-derived, the primary confirmatory detector is a custom enhanced residual
CNN, and this secondary detector uses SRNet architecture. All models remain
strictly disjoint from TEST during fitting and selection.

## 5. Training schedule

Training stegos use deterministic random allocation with the frozen reversible
codec.

1. curriculum warm-up: 0.012 net bpp, 2 epochs, learning rate 1e-3;
2. target training: 0.009 net bpp, 7 epochs, learning rate 1e-3;
3. target refinement: 0.009 net bpp, 4 epochs, learning rate 1e-4.

Each epoch samples 3,000 source pairs without replacement from the corresponding
fitting pool. The real minibatch contains four cover-stego pairs (eight images),
with identical dihedral augmentation within every pair. No gradient
accumulation is used.

The best checkpoint is selected **only** among the final low-learning-rate
target epochs using development ROC-AUC.

## 6. Resumability

A last-epoch checkpoint is written after every completed epoch and includes the
optimizer state, training history, NumPy generator state, and the pre-test
protocol SHA-256. Resumption is rejected when the protocol hash differs.

## 7. Pre-test gate

The selected checkpoint is evaluated on development at 0.009 net bpp. TEST
scoring is permitted only if

\[
\mathrm{AUC}_{dev}\ge 0.65.
\]

Failure terminates the notebook before any SRNet-v11 TEST inference. A failed
attempt must be preserved and reported as detector development, not silently
discarded.

## 8. TEST analysis after a successful gate

Only the frozen 0.009-net-bpp primary operating point is scored. The same
common-feasible source IDs are used for joint and predictability allocation.
Before inference, regenerated stegos are checked against frozen notebook-06
reversibility, net-payload, and PSNR invariants.

Report ROC-AUC, TPR at 5% FPR, minimal detection error under equal priors,
and source-image paired bootstrap differences.

## Reference

M. Boroumand, M. Chen, J. Fridrich, “Deep Residual Network for Steganalysis
of Digital Images,” IEEE Transactions on Information Forensics and Security,
vol. 14, no. 5, pp. 1181–1193, 2019. DOI: 10.1109/TIFS.2018.2871749.
