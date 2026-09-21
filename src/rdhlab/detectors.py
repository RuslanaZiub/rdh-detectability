from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import numpy as np
from scipy.ndimage import convolve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _quantized_residual_features(r: np.ndarray, trunc: int = 3) -> np.ndarray:
    q = np.clip(np.rint(r), -trunc, trunc).astype(np.int16) + trunc
    n = 2 * trunc + 1
    h = np.bincount(q.ravel(), minlength=n).astype(float)
    h /= max(h.sum(), 1.0)
    feats = [h]
    # Horizontal and vertical residual co-occurrences preserve local structure.
    for a, b in ((q[:, :-1], q[:, 1:]), (q[:-1, :], q[1:, :])):
        co = np.zeros((n, n), dtype=float)
        if a.size:
            np.add.at(co, (a.ravel(), b.ravel()), 1)
            co /= max(co.sum(), 1.0)
        feats.append(co.ravel())
    return np.concatenate(feats)


def srm_lite_features(image: np.ndarray, trunc: int = 3) -> np.ndarray:
    """SRM-inspired residual/co-occurrence descriptor.

    This is deliberately named *SRM-lite*: it is not the full Spatial Rich
    Model and must not be reported as such.
    """
    x = image.astype(np.float32)
    kernels = [
        np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=float),
        np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=float),
        np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float),
        np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]], dtype=float),
        np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=float),
        np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=float),
        np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=float),
    ]
    return np.concatenate([_quantized_residual_features(convolve(x, k, mode="reflect"), trunc) for k in kernels])


@dataclass
class RichDetector:
    model: Pipeline
    trunc: int = 3

    def score(self, image: np.ndarray) -> float:
        f = srm_lite_features(image, self.trunc).reshape(1, -1)
        return float(self.model.predict_proba(f)[0, 1])


def fit_rich_detector(covers: list[np.ndarray], stegos: list[np.ndarray], seed: int = 20260917, trunc: int = 3) -> RichDetector:
    if len(covers) != len(stegos):
        raise ValueError("covers/stegos length mismatch")
    X, y = [], []
    for c, s in zip(covers, stegos):
        X.append(srm_lite_features(c, trunc)); y.append(0)
        X.append(srm_lite_features(s, trunc)); y.append(1)
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, C=1.0, solver="liblinear", random_state=seed)),
    ])
    pipe.fit(np.stack(X), np.asarray(y))
    return RichDetector(pipe, trunc)


def detector_metrics(y_true: np.ndarray, scores: np.ndarray, fixed_fpr: float = 0.05) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    auc = float(roc_auc_score(y_true, scores))
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = np.flatnonzero(fpr <= fixed_fpr)
    tpr_at = float(tpr[valid[-1]]) if len(valid) else 0.0
    return {"auc": auc, "tpr_at_fpr": tpr_at, "fixed_fpr": float(fixed_fpr), "n": int(len(y_true))}


# ---- Compact residual CNN -------------------------------------------------

def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_residual_cnn():
    torch, nn = _torch()

    class ResidualCNN(nn.Module):
        def __init__(self):
            super().__init__()
            hp = torch.tensor([
                [[[-1., 2., -1.], [2., -4., 2.], [-1., 2., -1.]]],
                [[[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]],
                [[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]],
            ])
            self.hp = nn.Conv2d(1, 3, 3, padding=1, bias=False)
            with torch.no_grad():
                self.hp.weight.copy_(hp)
            self.hp.weight.requires_grad_(False)
            self.net = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.AvgPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.AvgPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AvgPool2d(2),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            )
            self.fc = nn.Linear(64, 1)

        def forward(self, x):
            x = self.hp(x)
            x = torch.clamp(x, -8.0, 8.0)
            x = self.net(x).flatten(1)
            return self.fc(x).squeeze(1)

    return ResidualCNN()


def train_residual_cnn(covers: list[np.ndarray], stegos: list[np.ndarray], validation: tuple[list[np.ndarray], list[np.ndarray]] | None = None, epochs: int = 5, batch_size: int = 16, lr: float = 1e-3, seed: int = 20260917, device: str | None = None):
    torch, nn = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_residual_cnn().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    arrays, labels = [], []
    for c, s in zip(covers, stegos):
        arrays += [c, s]; labels += [0.0, 1.0]
    idx = np.arange(len(arrays))
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        rng.shuffle(idx)
        model.train(); losses = []
        for start in range(0, len(idx), batch_size):
            ii = idx[start:start+batch_size]
            x = np.stack([arrays[i] for i in ii]).astype(np.float32) / 255.0
            y = np.asarray([labels[i] for i in ii], dtype=np.float32)
            xt = torch.from_numpy(x[:, None]).to(device)
            yt = torch.from_numpy(y).to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xt)
            loss = loss_fn(logits, yt)
            loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu()))
        rec = {"epoch": epoch + 1, "train_loss": float(np.mean(losses))}
        if validation is not None:
            vc, vs = validation
            ys = np.array([0] * len(vc) + [1] * len(vs))
            scores = score_residual_cnn(model, vc + vs, device=device, batch_size=batch_size)
            rec["val_auc"] = float(roc_auc_score(ys, scores))
        history.append(rec)
    return model, history, device


def score_residual_cnn(model, images: list[np.ndarray], device: str | None = None, batch_size: int = 32) -> np.ndarray:
    torch, _ = _torch()
    if device is None:
        device = next(model.parameters()).device.type
    model.eval(); scores = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            arr = np.stack(images[start:start+batch_size]).astype(np.float32) / 255.0
            xt = torch.from_numpy(arr[:, None]).to(device)
            p = torch.sigmoid(model(xt)).cpu().numpy()
            scores.extend(p.tolist())
    return np.asarray(scores, dtype=float)


def save_cnn(model, path: str | Path, metadata: dict | None = None):
    torch, _ = _torch()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    if metadata is not None:
        path.with_suffix(path.suffix + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_cnn(path: str | Path, device: str | None = None):
    torch, _ = _torch()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_residual_cnn().to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state); model.eval()
    return model, device


def paired_detector_bootstrap(cover_scores: np.ndarray, stego_scores: np.ndarray, fixed_fpr: float = 0.05, n_resamples: int = 2000, confidence: float = 0.95, seed: int = 20260917) -> dict:
    """Image-pair bootstrap for AUC and TPR@fixed FPR.

    Cover/stego observations from the same source image are resampled together,
    preserving the paired experimental unit.
    """
    c=np.asarray(cover_scores,dtype=float); s=np.asarray(stego_scores,dtype=float)
    if len(c)!=len(s): raise ValueError('cover/stego score length mismatch')
    n=len(c)
    if n==0: return {'auc_low':np.nan,'auc_high':np.nan,'tpr_low':np.nan,'tpr_high':np.nan,'n_pairs':0}
    rng=np.random.default_rng(seed)
    aucs=[]; tprs=[]
    for _ in range(n_resamples):
        idx=rng.integers(0,n,size=n)
        scores=np.column_stack([c[idx],s[idx]]).reshape(-1)
        y=np.tile([0,1],n)
        m=detector_metrics(y,scores,fixed_fpr)
        aucs.append(m['auc']); tprs.append(m['tpr_at_fpr'])
    a=(1-confidence)/2
    return {
        'auc_low':float(np.quantile(aucs,a)), 'auc_high':float(np.quantile(aucs,1-a)),
        'tpr_low':float(np.quantile(tprs,a)), 'tpr_high':float(np.quantile(tprs,1-a)),
        'n_pairs':int(n), 'bootstrap_resamples':int(n_resamples), 'confidence':float(confidence),
    }
