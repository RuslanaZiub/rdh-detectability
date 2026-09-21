# SRNet-v14 frozen TEST scoring protocol

## Role

Patch 14 performs the **post-hoc secondary robustness TEST evaluation** of the
SRNet-architecture detector developed and locked in Patch 13. It is not a new
primary endpoint and it must not be used to retune the RDH allocator, alpha,
payload, detector, checkpoint, or analysis rules.

## Locked detector

The patch ships an external locked copy of the exact Patch-13 selected checkpoint:

- payload: 0.009 net bpp
- selected target-stage epoch: 4
- DEV AUC: 0.8644054128159223
- DEV sanity gate: AUC >= 0.65
- checkpoint SHA-256:
  299dca8453fdd580d6cb169427cce67e5c98784df1f674c81172e327d791d640
- Patch-13 protocol SHA-256:
  41aa5f32cc783edf63f2db1c96c889ebd5bd3ea940fc55b54aaa1466a7f15685

The installer and notebook fail closed if the original Patch-13 artifacts do not
match these externally archived copies.

## TEST scope

Only the frozen primary scientific contrast is evaluated:

- payload: 0.009 net bpp
- methods: `joint` and `predictability`
- common-feasible source-image set from frozen notebook 06
- expected aligned pairs: 1968

The test covers are scored once. The two stego methods are regenerated from the
frozen notebook-06 contexts and cross-checked against the frozen notebook-06
records for exact reversibility, net payload, PSNR, SSIM, used blocks and changed
pixels before SRNet scoring.

## Metrics

For each method:
- ROC-AUC with paired-image bootstrap 95% CI
- TPR at 5% FPR with paired-image bootstrap 95% CI
- minimal detection error Pe with paired-image bootstrap 95% CI
- mean and median stego-minus-cover SRNet score change

Paired `joint - predictability` comparisons:
- mean score-change difference with 5000-bootstrap CI
- median score-change difference with 5000-bootstrap CI
- AUC difference with 5000-bootstrap CI
- TPR@5%FPR difference with 5000-bootstrap CI
- Pe difference with 5000-bootstrap CI

For score change, AUC and TPR, a negative difference indicates lower detectability
for `joint`. For Pe, a positive difference indicates lower detectability for `joint`.

## One fixed protocol, resumable execution

A TEST-access marker is written before scoring starts. If the run is interrupted,
the same immutable protocol may resume from prefix-validated CSV checkpoints.
A completed Patch-14 run is not rescored by `Run All`; existing complete artifacts
are loaded and displayed instead.

No training, checkpoint selection, allocator tuning or endpoint modification is
implemented in Patch 14.
