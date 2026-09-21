# SRNet-v13 TRAIN/DEV-only progressive curriculum protocol

## Purpose

Patch 13 is a detector-development experiment only. It does **not** score TEST.
It follows two failed pre-test SRNet development attempts and the TRAIN-only
payload-feasibility probe in Patch 12.

Patch 12 found exact recovery for every feasible probe case and the following
TRAIN-only feasibility fractions: 0.012 -> 0.958, 0.015 -> 0.924,
0.020 -> 0.868, 0.030 -> 0.702, and 0.050 -> 0.478. The pre-specified
broad-feasibility rule selected 0.015 bpp. Patch 13 preserves 0.015 bpp as the
bridge payload, while 0.050 and 0.030 bpp are used only as detector-bootstrap
payloads after the earlier 0.012->0.009 warm start failed to move DEV AUC away
from chance.

This is transparent post-hoc detector development on TRAIN/DEV only. It does
not change the frozen RDH allocator, alpha=0.25, the primary 0.009-bpp endpoint,
or any already reported TEST result.

## Data split

- SRNet fit pool: TRAIN[0:5250]
- SRNet development pool: TRAIN[5250:6000]
- TEST is not loaded for scoring and is never passed to the training helper.

## Progressive detector curriculum

The best development-AUC checkpoint at each stage seeds the next stage:

1. 0.050 bpp: random allocation, LR=1e-3, 3-6 epochs, 2200 pairs/epoch.
2. 0.030 bpp: random allocation, LR=1e-4, 2-5 epochs, 3000 pairs/epoch.
3. 0.015 bpp: random allocation, LR=1e-4, 2-5 epochs, 4000 pairs/epoch.
4. 0.009 bpp: random allocation, LR=1e-4, 4-8 epochs, 4500 pairs/epoch.

Real minibatch: 6 cover-stego pairs (12 images). Identical random dihedral
augmentation is applied to each aligned cover-stego pair.

The 0.050-bpp bootstrap stage has a CPU-saving progress gate: after 4 epochs,
if the best same-payload DEV AUC is still <0.55, the run halts and TEST remains
untouched. This is not a publication endpoint; it is a detector-development
sanity check.

The final pre-test gate remains:

    DEV AUC at 0.009 bpp >= 0.65

Only if that gate passes should a later patch be created to score the frozen
TEST split. Patch 13 intentionally contains no TEST-scoring cells.

## Relation to published SRNet

The network topology is the SRNet architecture used in earlier project patches.
The published SRNet training regime used substantially larger minibatches,
much larger training data, and tens to hundreds of thousands of updates. The
CPU-oriented protocol here is therefore a pragmatic SRNet-architecture
confirmatory detector-development procedure, not a reproduction of the
published training schedule.
