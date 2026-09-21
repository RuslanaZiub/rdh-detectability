# SRNet secondary robustness protocol

## Status

This analysis is **post-hoc secondary robustness evidence**. The original enhanced-CNN TEST results were already known before this analysis was designed. Consequently, SRNet results must not be described as a pre-specified primary endpoint.

## Frozen RDH quantities

No modification is permitted to:

\[
\alpha=0.25,
\qquad
R^*_{net}=0.009\ \mathrm{bpp},
\]

the SRM-derived teacher, the joint/predictability allocation rules, payload accounting, test source IDs, or the already frozen notebook-06 outputs.

## Detector independence

The local risk used by the allocator is SRM-derived. The existing primary confirmatory detector is a custom enhanced residual CNN. Patch 10 adds an independently trained **SRNet-architecture** detector whose convolutional filters are learned end-to-end and are not initialized with fixed SRM filters.

SRNet development uses only the TRAIN split:

- fitting: TRAIN indices 2750–5249 (2500 source images);
- detector development: TRAIN indices 5250–5999 (750 source images);
- embedding source for detector training: deterministic `random` allocator;
- payload: 0.009 net bpp.

These source images are disjoint from the enhanced-CNN fitting/development partition used by patch 05e.

## Pre-test gate

The detector protocol and checkpoint are hashed before SRNet scores TEST. A minimum detector-development ROC-AUC of 0.65 is required. Failure stops TEST scoring. Detector-development changes, if scientifically justified, must remain inside TRAIN/development data and create a new pre-test protocol hash. They must never modify the RDH allocator.

## TEST analysis

Only the frozen primary payload is used. The common-feasible source-image set is identical for `joint` and `predictability`. Before SRNet scoring, every regenerated stego is checked for:

\[
\mathrm{PER}_{cover}=0,
\qquad
\mathrm{BER}_{message}=0,
\]

and its net payload and PSNR are cross-checked against frozen notebook-06 results.

Report:

\[
\mathrm{ROC\!\! -\! AUC},
\qquad
\mathrm{TPR}@\mathrm{FPR}=0.05,
\qquad
P_E=\min_{\tau}\frac{P_{FA}(\tau)+P_{MD}(\tau)}{2}.
\]

Method differences are estimated with a source-image paired bootstrap.

## Reference

M. Boroumand, M. Chen, J. Fridrich, “Deep Residual Network for Steganalysis of Digital Images,” *IEEE Transactions on Information Forensics and Security*, vol. 14, no. 5, pp. 1181–1193, 2019. DOI: 10.1109/TIFS.2018.2871749.
