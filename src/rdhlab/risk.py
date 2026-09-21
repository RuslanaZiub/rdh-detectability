from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .blocks import iter_blocks
from .codec import capacity_bits, embed
from .features import block_features, predictability_score


@dataclass
class BlockRiskModel:
    model: Pipeline
    block_size: int
    probe_fraction: float
    probe_max_bits: int
    seed: int

    def risk_delta(self, block: np.ndarray, key: str = "") -> float:
        cap = capacity_bits(block)
        n = max(1, min(int(round(cap * self.probe_fraction)), self.probe_max_bits, cap))
        digest = hashlib.sha256(f"{self.seed}|{key}".encode()).digest()
        local_seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(local_seed)
        bits = rng.integers(0, 2, size=n, dtype=np.uint8)
        stego, _ = embed(block, bits)
        f0 = block_features(block).reshape(1, -1)
        f1 = block_features(stego).reshape(1, -1)
        d0 = float(self.model.decision_function(f0)[0])
        d1 = float(self.model.decision_function(f1)[0])
        return d1 - d0

    def score_image_blocks(self, image: np.ndarray, image_key: str = "") -> list[dict]:
        out = []
        for bid, y, x, block in iter_blocks(image, self.block_size):
            out.append({
                "block_id": int(bid), "y": int(y), "x": int(x),
                "predictability": float(predictability_score(block)),
                "detectability_risk": float(self.risk_delta(block, f"{image_key}:{bid}")),
            })
        return out


def fit_block_risk_model(images: list[np.ndarray], block_size: int = 64, probe_fraction: float = 0.25, probe_max_bits: int = 256, seed: int = 20260917, max_blocks: int | None = 50000) -> BlockRiskModel:
    X: list[np.ndarray] = []
    y: list[int] = []
    rng = np.random.default_rng(seed)
    count = 0
    for img_idx, image in enumerate(images):
        for bid, _, _, block in iter_blocks(image, block_size):
            cap = capacity_bits(block)
            n = max(1, min(int(round(cap * probe_fraction)), probe_max_bits, cap))
            bits = rng.integers(0, 2, size=n, dtype=np.uint8)
            stego, _ = embed(block, bits)
            X.append(block_features(block)); y.append(0)
            X.append(block_features(stego)); y.append(1)
            count += 1
            if max_blocks is not None and count >= max_blocks:
                break
        if max_blocks is not None and count >= max_blocks:
            break
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, C=1.0, solver="liblinear", random_state=seed)),
    ])
    pipe.fit(np.stack(X), np.asarray(y))
    return BlockRiskModel(pipe, block_size, probe_fraction, probe_max_bits, seed)
