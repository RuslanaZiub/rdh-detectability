# Patch 16 — reviewer-facing diagnostic audit

Status: descriptive audit of already-frozen artifacts.

No model is trained. No image is embedded. No detector is scored. No allocator
parameter is changed.

## Probe-size audit
The frozen notebook-06 context cache stores the exact `block_rows` used for the
TEST experiment. Each row includes `probe_bits` and `detectability_risk`.
Patch 16 aggregates all 2,000 TEST sources (16 x 64x64 blocks per image) to
characterize the realized distribution of n_probe and the frequency of very small
probes.

## Alpha-rule threshold audit
The validation-only `alpha_validation_summary.csv` observed before independent-CNN
transfer is reused. The original selection rule is recomputed at retention
thresholds 0.80, 0.90, and 0.95:
- choose the smallest interior alpha,
- retain at least tau of the alpha=0 vs alpha=1 median teacher-logit gain,
- require mean PSNR > the alpha=0 endpoint.

This sensitivity check does not access TEST and cannot replace the frozen alpha=0.25.
