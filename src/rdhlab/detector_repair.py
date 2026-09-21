from __future__ import annotations

from pathlib import Path
import json
import copy
import numpy as np

from .detectors import detector_metrics


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def _highpass_bank():
    """Eight zero-sum 5x5 residual kernels in raw pixel units."""
    k = np.zeros((8, 1, 5, 5), dtype=np.float32)
    # first-order directional residuals
    k[0, 0, 2, 1:3] = [-1, 1]
    k[1, 0, 1:3, 2] = [-1, 1]
    k[2, 0, 1, 1] = -1; k[2, 0, 2, 2] = 1
    k[3, 0, 1, 3] = -1; k[3, 0, 2, 2] = 1
    # second-order residuals
    k[4, 0, 2, 1:4] = [1, -2, 1]
    k[5, 0, 1:4, 2] = [1, -2, 1]
    # cross and diagonal Laplacians
    k[6, 0, 2, 2] = 4
    k[6, 0, 1, 2] = k[6, 0, 3, 2] = k[6, 0, 2, 1] = k[6, 0, 2, 3] = -1
    k[7, 0, 2, 2] = 4
    k[7, 0, 1, 1] = k[7, 0, 1, 3] = k[7, 0, 3, 1] = k[7, 0, 3, 3] = -1
    return k


def build_enhanced_residual_cnn():
    """Compact steganalysis CNN that preserves +/-1 pixel evidence.

    Unlike the v0.2.0 baseline CNN, the fixed high-pass bank receives images in
    raw 0..255 pixel units rather than after division by 255. This avoids
    attenuating an embedding change of one gray level to 1/255 before residual
    extraction. The architecture is intentionally compact and is not claimed to
    be SRNet/XuNet.
    """
    torch, nn = _torch()

    class EnhancedResidualCNN(nn.Module):
        def __init__(self):
            super().__init__()
            bank = torch.from_numpy(_highpass_bank())
            self.pad = nn.ReflectionPad2d(2)
            self.hp = nn.Conv2d(1, 8, 5, padding=0, bias=False)
            with torch.no_grad():
                self.hp.weight.copy_(bank)
            self.hp.weight.requires_grad_(False)
            self.features = nn.Sequential(
                nn.Conv2d(8, 24, 3, padding=1, bias=False),
                nn.BatchNorm2d(24), nn.Tanh(),
                nn.Conv2d(24, 24, 3, padding=1, bias=False),
                nn.BatchNorm2d(24), nn.ReLU(inplace=True), nn.AvgPool2d(2),
                nn.Conv2d(24, 48, 3, padding=1, bias=False),
                nn.BatchNorm2d(48), nn.ReLU(inplace=True), nn.AvgPool2d(2),
                nn.Conv2d(48, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.AvgPool2d(2),
                nn.Conv2d(64, 96, 3, padding=1, bias=False),
                nn.BatchNorm2d(96), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.fc = nn.Linear(96, 1)

        def forward(self, x):
            # x is deliberately raw grayscale intensity (0..255).
            x = self.hp(self.pad(x))
            x = torch.clamp(x, -4.0, 4.0)
            x = self.features(x).flatten(1)
            return self.fc(x).squeeze(1)

    return EnhancedResidualCNN()


def _dihedral_pair(c: np.ndarray, s: np.ndarray, code: int):
    """Apply the same deterministic dihedral transform to a cover/stego pair."""
    k = int(code) % 4
    c2 = np.rot90(c, k)
    s2 = np.rot90(s, k)
    if int(code) >= 4:
        c2 = np.fliplr(c2)
        s2 = np.fliplr(s2)
    return np.ascontiguousarray(c2), np.ascontiguousarray(s2)


def score_enhanced_residual_cnn(model, images, *, device: str | None = None, batch_size: int = 16):
    torch, _ = _torch()
    if device is None:
        device = next(model.parameters()).device.type
    model.eval(); out = []
    with torch.no_grad():
        for start in range(0, len(images), int(batch_size)):
            arr = np.stack(images[start:start+int(batch_size)]).astype(np.float32)
            xt = torch.from_numpy(arr[:, None]).to(device)
            out.extend(torch.sigmoid(model(xt)).cpu().numpy().tolist())
    return np.asarray(out, dtype=float)


def train_enhanced_residual_cnn(
    covers,
    stegos,
    *,
    validation=None,
    epochs: int = 6,
    pair_batch_size: int = 8,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 20260917,
    device: str | None = None,
    fixed_fpr: float = 0.05,
):
    """Train with cover/stego-paired batches and paired dihedral augmentation.

    Epoch count and optimizer settings are fixed by the caller. Validation is
    diagnostic only: no early stopping or best-epoch selection is performed.
    The returned model is always the final fixed epoch.
    """
    if len(covers) != len(stegos):
        raise ValueError("covers/stegos length mismatch")
    if len(covers) == 0:
        raise ValueError("empty training set")

    torch, nn = _torch()
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_enhanced_residual_cnn().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(int(seed))
    pair_ids = np.arange(len(covers))
    history = []

    for epoch in range(int(epochs)):
        rng.shuffle(pair_ids)
        model.train(); losses = []
        for start in range(0, len(pair_ids), int(pair_batch_size)):
            ids = pair_ids[start:start+int(pair_batch_size)]
            xs, ys = [], []
            for i in ids:
                code = int(rng.integers(0, 8))
                c, s = _dihedral_pair(covers[int(i)], stegos[int(i)], code)
                xs.extend([c, s]); ys.extend([0.0, 1.0])
            arr = np.stack(xs).astype(np.float32)
            xt = torch.from_numpy(arr[:, None]).to(device)
            yt = torch.tensor(ys, dtype=torch.float32, device=device)
            opt.zero_grad(set_to_none=True)
            logits = model(xt)
            loss = loss_fn(logits, yt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        rec = {"epoch": epoch + 1, "train_loss": float(np.mean(losses))}
        if validation is not None:
            vc, vs = validation
            cs = score_enhanced_residual_cnn(model, vc, device=device, batch_size=2*int(pair_batch_size))
            ss = score_enhanced_residual_cnn(model, vs, device=device, batch_size=2*int(pair_batch_size))
            y = np.tile([0, 1], len(vc))
            scores = np.column_stack([cs, ss]).reshape(-1)
            met = detector_metrics(y, scores, fixed_fpr=float(fixed_fpr))
            rec.update({
                "dev_auc": float(met["auc"]),
                "dev_tpr_at_fpr": float(met["tpr_at_fpr"]),
                "dev_cover_score_mean": float(np.mean(cs)),
                "dev_stego_score_mean": float(np.mean(ss)),
            })
        history.append(rec)
    return model, history, device


def save_enhanced_cnn(model, path: str | Path, metadata: dict | None = None):
    torch, _ = _torch()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    if metadata is not None:
        path.with_suffix(path.suffix + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_enhanced_cnn(path: str | Path, device: str | None = None):
    torch, _ = _torch()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_enhanced_residual_cnn().to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state); model.eval()
    return model, device
