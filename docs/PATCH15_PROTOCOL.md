# Patch 15 — post-hoc reviewer analyses

This patch is intentionally **post-hoc and exploratory**. It is created after the
primary TEST result and after the SRNet secondary TEST result have been observed.
It MUST NOT be used to change alpha, the primary endpoint, the detector, the payload
grid, or any frozen result.

It addresses two reviewer-facing diagnostics:

1. **Alpha sensitivity at the frozen primary payload (0.009 net bpp)** for
   alpha in {0, 0.25, 0.5, 0.75, 1}. The endpoints have an exact interpretation:
   alpha=0 is detectability-only ranking and alpha=1 is predictability-only ranking.
   Alpha=0.25 is the already-frozen joint allocator. Alpha=0.5 and 0.75 are new,
   post-hoc sensitivity points.

2. **A controlled hand-crafted complexity baseline** under the same reversible
   codec, same TEST sources and same net payload. The baseline ranks 64x64 blocks
   by descending mean Sobel gradient magnitude:
       C_i = mean(|G_x|) + mean(|G_y|).
   This is NOT claimed to reproduce Hong et al. or Chen et al.; it is a same-codec
   complexity-only control designed to test whether the detector-derived signal
   contributes beyond a conventional local-activity proxy.

All new stego cases are checked for exact image recovery, exact message recovery,
zero BER and exact net-payload accounting before frozen enhanced-CNN scoring.
The frozen cover scores and frozen enhanced-CNN weights from the primary analysis
are reused unchanged. Bootstrap comparisons are paired by source image.
