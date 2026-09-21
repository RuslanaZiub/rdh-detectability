# SRNet-v12 TRAIN-only curriculum payload feasibility probe

## 1. Purpose

Patch 11 remained at approximately random development discrimination after three epochs (`AUC ≈ 0.50`) and was interrupted before TEST. Patch 12 therefore does **not** launch another long SRNet training run. It first determines whether the frozen reversible codec can provide a substantially stronger TRAIN-only stego signal for curriculum pretraining.

This patch is detector-development diagnostics only. It does not alter the RDH method or any publication endpoint.

## 2. Frozen quantities

The following remain unchanged:

\[
\alpha=0.25,\qquad R^*_{net}=0.009\ \mathrm{bpp}.
\]

No modification is permitted to the SRM-derived local-risk teacher, joint/predictability allocation rules, frozen TEST source IDs, four publication payloads, notebook-06 TEST results, or enhanced-CNN primary confirmation.

## 3. Data isolation

The probe uses **500 source images sampled deterministically from the 5,250-image TRAIN fitting pool only**. The 750-image development partition and the 2,000-image TEST split are excluded.

No TEST image is read, embedded, scored, or used for payload selection.

## 4. Pre-specified payload grid

The frozen probe grid is

\[
R_{probe}\in\{0.012,0.015,0.020,0.030,0.050\}\ \mathrm{net\ bpp}.
\]

`0.012` bpp is the Patch-11 reference; the other rates are candidate stronger curriculum payloads.

For each source and payload, deterministic random block allocation is used with the same frozen reversible codec. The probe records feasibility, exact image/message recovery, actual net payload, distortion, side-information quantities when exposed by the pipeline, block usage, changed pixels, and runtime.

## 5. Selection rule

A candidate stronger curriculum payload is eligible only if:

1. it is strictly greater than `0.012` net bpp;
2. at least 90% of the 500 probe sources are feasible;
3. every feasible case has exact cover and message recovery.

Among eligible rates, the **largest pre-specified payload** is selected. This rule is fixed before running the probe.

If no candidate satisfies the rule, Patch 12 returns `NO_STRONG_PAYLOAD_FOUND` and no new SRNet training run should be started from this patch.

## 6. Scientific interpretation

The selected rate is a **detector-training curriculum rate only**. It must not be reported as a new RDH evaluation payload and must not be used to retune `alpha`, the allocator, or the primary endpoint.

Patch 13, if created, may use the selected TRAIN-only curriculum path while keeping the pre-test SRNet gate at development AUC >= 0.65.
