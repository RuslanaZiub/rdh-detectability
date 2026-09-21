from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Iterable

import numpy as np

from .blocks import iter_blocks
from .codec import HSMetadata, capacity_bits, embed, extract, location_map_storage_bits

# Accounting model for a compact future self-contained representation.
# These bits are not physically embedded in v0.2.0; they are subtracted from
# gross payload so all methods are compared at equal accounted net payload.
GLOBAL_HEADER_BITS = 64  # compact image-level version/method/payload/block-count header
LOCATION_MAP_LENGTH_BITS = 16


@dataclass(frozen=True)
class BlockPlan:
    block_id: int
    y: int
    x: int
    capacity_bits: int
    sideinfo_bits: int
    net_gain_bits: int
    predictability: float | None = None
    detectability_risk: float | None = None


@dataclass(frozen=True)
class UsedBlock:
    block_id: int
    y: int
    x: int
    payload_bits: int
    metadata: HSMetadata


@dataclass(frozen=True)
class BlockwiseMetadata:
    block_size: int
    image_height: int
    image_width: int
    gross_payload_bits: int
    target_net_payload_bits: int
    accounted_sideinfo_bits: int
    used_blocks: tuple[UsedBlock, ...]


def compact_block_sideinfo_bits(meta: HSMetadata, block_pixels: int, total_blocks: int) -> int:
    """Compact binary accounting for one used block.

    Fields: block id, peak, target, payload length, location-map flag and,
    only when present, a compressed-map byte-length plus compressed bitmap.
    Direction is inferred from target versus peak. The engineering JSON is
    never charged as payload overhead.
    """
    block_id_bits=max(1,int(np.ceil(np.log2(max(2,total_blocks)))))
    payload_len_bits=max(1,int(np.ceil(np.log2(block_pixels+1))))
    base=block_id_bits + 8 + 8 + payload_len_bits + 1
    lm=location_map_storage_bits(meta)
    return int(base if lm==0 else base + LOCATION_MAP_LENGTH_BITS + lm)


def _block_sideinfo_estimate(block: np.ndarray, total_blocks: int) -> int:
    _, meta = embed(block, np.empty(0, dtype=np.uint8))
    return compact_block_sideinfo_bits(meta, int(block.size), int(total_blocks))


def analyze_blocks(image: np.ndarray, block_size: int = 64) -> list[BlockPlan]:
    raw=list(iter_blocks(image, block_size))
    total_blocks=len(raw)
    plans: list[BlockPlan] = []
    for bid, y, x, block in raw:
        cap = int(capacity_bits(block))
        overhead = int(_block_sideinfo_estimate(block,total_blocks))
        plans.append(
            BlockPlan(
                block_id=int(bid),
                y=int(y),
                x=int(x),
                capacity_bits=cap,
                sideinfo_bits=overhead,
                net_gain_bits=cap - overhead,
            )
        )
    return plans


def max_accounted_net_capacity(image: np.ndarray, block_size: int = 64) -> dict:
    plans = analyze_blocks(image, block_size)
    eligible = [p for p in plans if p.net_gain_bits > 0]
    gross = sum(p.capacity_bits for p in eligible)
    side = GLOBAL_HEADER_BITS + sum(p.sideinfo_bits for p in eligible)
    net = max(0, gross - side)
    return {
        "gross_capacity_bits": int(gross),
        "sideinfo_bits": int(side),
        "max_net_payload_bits": int(net),
        "max_net_bpp": float(net / image.size),
        "eligible_blocks": int(len(eligible)),
        "total_blocks": int(len(plans)),
    }


def _select_for_target(plans_by_id: dict[int, BlockPlan], order: Iterable[int], target_net_bits: int) -> tuple[list[BlockPlan], int]:
    if target_net_bits < 0:
        raise ValueError("target_net_bits must be non-negative")
    selected: list[BlockPlan] = []
    gross_cap = 0
    side = GLOBAL_HEADER_BITS
    seen: set[int] = set()
    for bid_raw in order:
        bid = int(bid_raw)
        if bid in seen or bid not in plans_by_id:
            continue
        seen.add(bid)
        p = plans_by_id[bid]
        if p.net_gain_bits <= 0:
            continue
        selected.append(p)
        gross_cap += p.capacity_bits
        side += p.sideinfo_bits
        if gross_cap - side >= target_net_bits:
            return selected, side
    raise ValueError(
        f"Target net payload {target_net_bits} bits is infeasible for the supplied block order; "
        f"maximum reached was {max(0, gross_cap-side)} bits"
    )


def embed_blockwise(
    image: np.ndarray,
    payload_bits: np.ndarray | list[int],
    order: Iterable[int],
    block_size: int = 64,
    plans: list[BlockPlan] | None = None,
) -> tuple[np.ndarray, BlockwiseMetadata, dict]:
    """Embed an accounted-net payload with independently reversible blocks.

    ``payload_bits`` are the *useful* payload bits. Additional random-looking
    filler bits are supplied internally by the caller through ``gross_bits`` in
    ``embed_blockwise_with_rng``. This low-level function instead treats the
    provided array as gross payload and reports its accounted net value.
    """
    image = np.asarray(image, dtype=np.uint8)
    bits = np.asarray(payload_bits, dtype=np.uint8).ravel()
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("payload_bits must contain only 0/1")

    plans = analyze_blocks(image, block_size) if plans is None else plans
    by_id = {p.block_id: p for p in plans}
    # Here len(bits) is gross; select enough blocks to hold it, not by net.
    selected: list[BlockPlan] = []
    gross_cap = 0
    seen: set[int] = set()
    for bid_raw in order:
        bid = int(bid_raw)
        if bid in seen or bid not in by_id:
            continue
        seen.add(bid)
        p = by_id[bid]
        if p.net_gain_bits <= 0:
            continue
        selected.append(p)
        gross_cap += p.capacity_bits
        if gross_cap >= len(bits):
            break
    if gross_cap < len(bits):
        raise ValueError(f"Gross payload {len(bits)} exceeds ordered positive-net block capacity {gross_cap}")

    out = image.copy()
    used: list[UsedBlock] = []
    offset = 0
    t0 = perf_counter_ns()
    for p in selected:
        if offset >= len(bits):
            break
        n = min(p.capacity_bits, len(bits) - offset)
        block = out[p.y : p.y + block_size, p.x : p.x + block_size]
        stego_block, meta = embed(block, bits[offset : offset + n])
        out[p.y : p.y + block_size, p.x : p.x + block_size] = stego_block
        used.append(UsedBlock(p.block_id, p.y, p.x, int(n), meta))
        offset += n
    encode_ms = (perf_counter_ns() - t0) / 1e6

    total_blocks=len(plans)
    side = GLOBAL_HEADER_BITS + sum(compact_block_sideinfo_bits(u.metadata, block_size*block_size, total_blocks) for u in used)
    meta = BlockwiseMetadata(
        block_size=int(block_size),
        image_height=int(image.shape[0]),
        image_width=int(image.shape[1]),
        gross_payload_bits=int(len(bits)),
        target_net_payload_bits=int(max(0, len(bits) - side)),
        accounted_sideinfo_bits=int(side),
        used_blocks=tuple(used),
    )
    stats = {
        "encode_ms": float(encode_ms),
        "used_blocks": int(len(used)),
        "eligible_blocks": int(sum(p.net_gain_bits > 0 for p in plans)),
        "total_blocks": int(len(plans)),
        "accounted_sideinfo_bits": int(side),
        "accounted_net_payload_bits": int(len(bits) - side),
    }
    return out, meta, stats


def embed_blockwise_net(
    image: np.ndarray,
    target_net_bits: int,
    order: Iterable[int],
    rng: np.random.Generator,
    block_size: int = 64,
    plans: list[BlockPlan] | None = None,
) -> tuple[np.ndarray, np.ndarray, BlockwiseMetadata, dict]:
    """Embed exactly ``target_net_bits`` after compact side-info accounting."""
    image = np.asarray(image, dtype=np.uint8)
    plans = analyze_blocks(image, block_size) if plans is None else plans
    by_id = {p.block_id: p for p in plans}
    selected, side = _select_for_target(by_id, order, int(target_net_bits))
    gross_needed = int(target_net_bits) + int(side)
    gross_capacity = sum(p.capacity_bits for p in selected)
    if gross_needed > gross_capacity:
        raise AssertionError("internal target selection error")
    gross_bits = rng.integers(0, 2, size=gross_needed, dtype=np.uint8)
    # Limit order to selected ids so the same overhead model is realized.
    selected_order = [p.block_id for p in selected]
    stego, meta, stats = embed_blockwise(image, gross_bits, selected_order, block_size, plans=plans)
    if meta.accounted_sideinfo_bits != side:
        raise AssertionError("side-information estimate changed during embedding")
    if meta.gross_payload_bits - meta.accounted_sideinfo_bits != int(target_net_bits):
        raise AssertionError("net payload accounting mismatch")
    meta = BlockwiseMetadata(
        block_size=meta.block_size,
        image_height=meta.image_height,
        image_width=meta.image_width,
        gross_payload_bits=meta.gross_payload_bits,
        target_net_payload_bits=int(target_net_bits),
        accounted_sideinfo_bits=meta.accounted_sideinfo_bits,
        used_blocks=meta.used_blocks,
    )
    stats["accounted_net_payload_bits"] = int(target_net_bits)
    stats["gross_capacity_selected_bits"] = int(gross_capacity)
    return stego, gross_bits, meta, stats


def extract_blockwise(stego: np.ndarray, meta: BlockwiseMetadata) -> tuple[np.ndarray, np.ndarray, dict]:
    y = np.asarray(stego, dtype=np.uint8).copy()
    parts: list[np.ndarray] = []
    t0 = perf_counter_ns()
    for u in meta.used_blocks:
        block = y[u.y : u.y + meta.block_size, u.x : u.x + meta.block_size]
        recovered, bits = extract(block, u.metadata)
        y[u.y : u.y + meta.block_size, u.x : u.x + meta.block_size] = recovered
        parts.append(bits)
    decode_ms = (perf_counter_ns() - t0) / 1e6
    payload = np.concatenate(parts) if parts else np.empty(0, dtype=np.uint8)
    payload = payload[: meta.gross_payload_bits]
    return y, payload, {"decode_ms": float(decode_ms)}
