from __future__ import annotations
from dataclasses import dataclass, asdict
import base64
import json
import zlib
import numpy as np


@dataclass(frozen=True)
class HSMetadata:
    peak: int
    zero: int
    payload_bits: int
    direction: int
    # When the selected target bin is not empty, original pixels already
    # occupying that bin are represented by a compressed location map.
    # The map is stored in the engineering sidecar and its compressed binary
    # size is included in side-information accounting.
    location_map_b64: str | None = None
    location_map_nbits: int = 0
    location_map_count: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_json(text: str) -> "HSMetadata":
        return HSMetadata(**json.loads(text))


def _hist(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("image must be uint8 grayscale")
    return np.bincount(image.ravel(), minlength=256)


def _target_for_peak(h: np.ndarray, p: int) -> tuple[int, int]:
    """Return a deterministic target bin and direction for peak p.

    Prefer a true empty bin because no location map is then required. If the
    histogram has full 8-bit support, choose a least-populated non-peak bin;
    the codec will preserve the pre-existing target-bin pixels via a location
    map. Ties are broken by the shortest shift distance, then by bin value.
    """
    candidates = np.arange(256, dtype=int)
    candidates = candidates[candidates != p]

    empty = candidates[h[candidates] == 0]
    pool = empty if len(empty) else candidates

    # Lexicographic objective: occupancy first, then |z-p|, then z.
    occupancy = h[pool]
    distance = np.abs(pool - p)
    order = np.lexsort((pool, distance, occupancy))
    z = int(pool[order[0]])
    direction = +1 if z > p else -1
    return z, direction


def choose_peak_zero(image: np.ndarray) -> tuple[int, int, int]:
    """Choose a histogram peak and target bin for reversible HS.

    Historical name retained for API compatibility. A true zero bin is used
    whenever one exists. For full-support histograms (common after JPEG
    decoding), the least-populated target bin is used together with a
    compressed location map, so exact reversibility is still possible.
    """
    h = _hist(image)
    max_count = int(h.max())
    peaks = np.flatnonzero(h == max_count)

    best: tuple[int, int, int, int, int] | None = None
    # score = (target occupancy, shift distance, peak value, target value, direction)
    for p_raw in peaks:
        p = int(p_raw)
        z, direction = _target_for_peak(h, p)
        score = (int(h[z]), abs(z - p), p, z, direction)
        if best is None or score[:4] < best[:4]:
            best = score

    assert best is not None
    _, _, p, z, direction = best
    return int(p), int(z), int(direction)


def capacity_bits(image: np.ndarray, peak: int | None = None) -> int:
    h = _hist(image)
    if peak is None:
        peak, _, _ = choose_peak_zero(image)
    return int(h[peak])


def _encode_location_map(mask: np.ndarray) -> str | None:
    mask = np.asarray(mask, dtype=np.uint8).ravel()
    if not np.any(mask):
        return None
    packed = np.packbits(mask, bitorder="little").tobytes()
    compressed = zlib.compress(packed, level=9)
    return base64.b64encode(compressed).decode("ascii")


def _decode_location_map(meta: HSMetadata, size: int) -> np.ndarray:
    if not meta.location_map_b64:
        return np.zeros(size, dtype=bool)
    if meta.location_map_nbits != size:
        raise ValueError(
            f"Location-map size mismatch: metadata={meta.location_map_nbits}, image={size}"
        )
    compressed = base64.b64decode(meta.location_map_b64.encode("ascii"))
    packed = zlib.decompress(compressed)
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")
    if len(bits) < size:
        raise ValueError("Corrupt location map: not enough bits")
    return bits[:size].astype(bool)


def location_map_storage_bits(meta: HSMetadata) -> int:
    """Binary storage needed for the compressed location map only."""
    if not meta.location_map_b64:
        return 0
    compressed = base64.b64decode(meta.location_map_b64.encode("ascii"))
    return len(compressed) * 8


def embed(image: np.ndarray, bits: np.ndarray | list[int]) -> tuple[np.ndarray, HSMetadata]:
    image = np.asarray(image, dtype=np.uint8)
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("bits must contain only 0/1")

    p, z, direction = choose_peak_zero(image)
    cap = capacity_bits(image, p)
    if len(bits) > cap:
        raise ValueError(f"Payload too large: {len(bits)} > capacity {cap}")

    original_flat = image.ravel()
    location_mask = original_flat == z
    location_map_b64 = _encode_location_map(location_mask)

    x = image.astype(np.int16).copy()
    if direction > 0:
        # Preserve original z pixels; only the open interval is shifted.
        mask = (x > p) & (x < z)
        x[mask] += 1
    else:
        mask = (x < p) & (x > z)
        x[mask] -= 1

    positions = np.flatnonzero(x.ravel() == p)[: len(bits)]
    flat = x.ravel()
    flat[positions] = p + direction * bits.astype(np.int16)
    stego = flat.reshape(x.shape)
    if stego.min() < 0 or stego.max() > 255:
        raise AssertionError("Internal overflow in histogram shifting")

    meta = HSMetadata(
        peak=p,
        zero=z,
        payload_bits=int(len(bits)),
        direction=direction,
        location_map_b64=location_map_b64,
        location_map_nbits=int(image.size) if location_map_b64 else 0,
        location_map_count=int(location_mask.sum()),
    )
    return stego.astype(np.uint8), meta


def extract(stego: np.ndarray, meta: HSMetadata) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(stego, dtype=np.uint8).astype(np.int16).copy()
    p, z, direction, n = meta.peak, meta.zero, meta.direction, meta.payload_bits
    flat = y.ravel()
    location_mask = _decode_location_map(meta, flat.size)

    if direction > 0:
        candidate_mask = (flat == p) | (flat == p + 1)
        # If z == p+1, pre-existing z pixels collide with embedded bit-1
        # symbols. Exclude them using the stored location map.
        if z == p + 1:
            candidate_mask &= ~location_mask
        candidates = np.flatnonzero(candidate_mask)[:n]
        if len(candidates) != n:
            raise ValueError("Not enough embedded symbols found")
        bits = (flat[candidates] - p).astype(np.uint8)
        flat[candidates] = p

        restored = flat.reshape(y.shape)
        restore_mask = (restored > p + 1) & (restored <= z)
        if meta.location_map_b64:
            restore_mask &= ~location_mask.reshape(y.shape)
        restored[restore_mask] -= 1
    else:
        candidate_mask = (flat == p) | (flat == p - 1)
        if z == p - 1:
            candidate_mask &= ~location_mask
        candidates = np.flatnonzero(candidate_mask)[:n]
        if len(candidates) != n:
            raise ValueError("Not enough embedded symbols found")
        bits = (p - flat[candidates]).astype(np.uint8)
        flat[candidates] = p

        restored = flat.reshape(y.shape)
        restore_mask = (restored < p - 1) & (restored >= z)
        if meta.location_map_b64:
            restore_mask &= ~location_mask.reshape(y.shape)
        restored[restore_mask] += 1

    if restored.min() < 0 or restored.max() > 255:
        raise AssertionError("Internal restoration overflow")
    return restored.astype(np.uint8), bits


def sideinfo_bits(meta: HSMetadata) -> int:
    """Accounting estimate for a compact binary side-information header.

    Base header:
      peak 8 + target bin 8 + direction 1 + payload length 32 +
      location-map-present flag 1 = 50 bits.

    If a location map is required, add a 32-bit compressed-byte-length field
    and the compressed packed bitmap itself. JSON/base64 is only the current
    engineering sidecar representation and is *not* used for bit accounting.
    """
    base = 50
    lm = location_map_storage_bits(meta)
    return base if lm == 0 else base + 32 + lm
