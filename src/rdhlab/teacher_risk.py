from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np

from .blocks import iter_blocks
from .codec import capacity_bits, embed
from .detectors import RichDetector, srm_lite_features
from .features import predictability_score


def rich_detector_logit(detector: RichDetector, image: np.ndarray) -> float:
    """Return the pre-sigmoid/log-odds output of a fitted SRM-lite detector."""
    f = srm_lite_features(np.asarray(image, dtype=np.uint8), detector.trunc).reshape(1, -1)
    return float(detector.model.decision_function(f)[0])


@dataclass
class SRMTeacherLocalRisk:
    """Local marginal detectability risk induced by an image-level SRM-lite teacher.

    For each block, a deterministic reversible probe is embedded only in that block.
    Risk is the change in the teacher logit, optionally normalized by the number of
    embedded probe bits. The teacher must be trained only on the training split.
    """

    teacher: RichDetector
    block_size: int = 64
    probe_fraction: float = 0.25
    probe_max_bits: int = 128
    seed: int = 20260917
    normalize_by_bits: bool = True

    def _probe_bits(self, block: np.ndarray) -> int:
        cap = int(capacity_bits(block))
        if cap <= 0:
            return 0
        n = int(round(cap * float(self.probe_fraction)))
        return max(1, min(n, int(self.probe_max_bits), cap))

    def _rng(self, key: str) -> np.random.Generator:
        digest = hashlib.sha256(f"{self.seed}|{key}".encode()).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "little"))

    def risk_delta(self, image: np.ndarray, y: int, x: int, block_id: int, image_key: str = "", base_logit: float | None = None) -> tuple[float, int]:
        image = np.asarray(image, dtype=np.uint8)
        block = image[y:y+self.block_size, x:x+self.block_size]
        n = self._probe_bits(block)
        if n <= 0:
            return 0.0, 0
        rng = self._rng(f"{image_key}:{block_id}")
        bits = rng.integers(0, 2, size=n, dtype=np.uint8)
        stego_block, _ = embed(block, bits)
        probe = image.copy()
        probe[y:y+self.block_size, x:x+self.block_size] = stego_block
        if base_logit is None:
            base_logit = rich_detector_logit(self.teacher, image)
        delta = rich_detector_logit(self.teacher, probe) - float(base_logit)
        if self.normalize_by_bits:
            delta /= float(n)
        return float(delta), int(n)

    def score_image_blocks(self, image: np.ndarray, image_key: str = "") -> list[dict]:
        image = np.asarray(image, dtype=np.uint8)
        base = rich_detector_logit(self.teacher, image)
        out: list[dict] = []
        for bid, y, x, block in iter_blocks(image, self.block_size):
            d, n = self.risk_delta(image, y, x, bid, image_key=image_key, base_logit=base)
            out.append({
                "block_id": int(bid),
                "y": int(y),
                "x": int(x),
                "predictability": float(predictability_score(block)),
                "detectability_risk": float(d),
                "probe_bits": int(n),
                "teacher_base_logit": float(base),
            })
        return out
