# Research protocol v0.2.0

## 1. Data separation

The source split is created once and never reshuffled: 6000 train / 2000 validation / 2000 test. All derivatives of a source image, including QF variants, retain the same split.

## 2. Frozen v0.1.7 audit

The original global reversible codec is evaluated on all test images to establish exact recovery, gross/net accounting, distortion and runtime. These files remain separate from v0.2.0 method results.

## 3. Net-payload freeze

Candidate net-payload levels are declared before viewing test capacity. Feasibility is measured only on validation. If the declared 0.05/0.10/0.20/0.30 bpp grid is too high for the accounted blockwise codec, a lower grid is derived from the validation 5th-percentile capacity and frozen in JSON.

## 4. Auxiliary risk model

Train-only cover/probe-stego block pairs fit a logistic detector. `D_i` is the detector decision-function change produced by probe embedding. The model is an allocation instrument, not the final steganalyzer.

## 5. Hypothesis analysis

Validation and final test analyses report Spearman rho, Kendall tau and rank disagreement between predictability `P_i` and risk `D_i`. Image-cluster bootstrap is used because blocks from the same image are not independent.

## 6. Joint allocator freeze

`alpha` is chosen from a fixed grid on validation only. Test data do not influence the choice.

## 7. Frozen test ablation

At every frozen net payload, compare raster, random, predictability-only, detectability-only and joint allocation on all 2000 test images. Record per-image feasibility, gross payload, side information, net payload, PSNR, SSIM, exact recovery, BER, changed pixels, selected-block statistics, encode/decode latency.

## 8. Independent security evaluation

Final detectability uses detector classes not used in allocation: SRM-lite and a residual CNN. Report ROC-AUC and TPR at fixed FPR. Additionally retrain an adaptive SRM-lite detector per final strategy at the primary payload.

## 9. Robustness

QF75/QF85/QF95 preprocessing-history variants are evaluated only after the primary experiment is frozen and without retuning. A later canonical-lossless BOSSbase experiment should be added before final submission if available.

## 10. Claims discipline

Mathematical exact reversibility is distinguished from empirical detectability. Finite detector evidence does not establish universal steganographic security. Accounted side-information overhead is distinguished from a physically self-contained metadata channel.
