from __future__ import annotations

from pathlib import Path
import json
import random

import numpy as np


def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def build_srnet(num_classes: int = 2):
    """Build an SRNet-architecture steganalyzer for grayscale images.

    Topology follows Boroumand, Chen & Fridrich, IEEE TIFS 2019:
    2 Type-1 layers, 5 unpooled residual Type-2 layers, 4 pooled residual
    Type-3 layers, one Type-4 layer with global average pooling, and a linear
    classifier. All convolution kernels are learned end-to-end; no fixed SRM
    filters are used.

    The training schedule in this project is intentionally frozen and
    project-specific. It is not claimed to reproduce the original paper's
    500k-iteration training protocol.
    """
    torch, nn, F = _torch()

    class Type1(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)
            self.bn = nn.BatchNorm2d(out_ch, momentum=0.1)

        def forward(self, x):
            return F.relu(self.bn(self.conv(x)), inplace=False)

    class Type2(nn.Module):
        def __init__(self, ch: int):
            super().__init__()
            self.first = Type1(ch, ch)
            self.conv = nn.Conv2d(ch, ch, 3, padding=1, bias=True)
            self.bn = nn.BatchNorm2d(ch, momentum=0.1)

        def forward(self, x):
            return x + self.bn(self.conv(self.first(x)))

    class Type3(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.first = Type1(in_ch, out_ch)
            self.conv = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True)
            self.bn = nn.BatchNorm2d(out_ch, momentum=0.1)
            self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=2, bias=True)
            self.skip_bn = nn.BatchNorm2d(out_ch, momentum=0.1)

        def forward(self, x):
            y = self.bn(self.conv(self.first(x)))
            y = F.avg_pool2d(y, kernel_size=3, stride=2, padding=1)
            return y + self.skip_bn(self.skip(x))

    class Type4(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.first = Type1(in_ch, out_ch)
            self.conv = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True)
            self.bn = nn.BatchNorm2d(out_ch, momentum=0.1)

        def forward(self, x):
            x = self.bn(self.conv(self.first(x)))
            return F.adaptive_avg_pool2d(x, 1).flatten(1)

    class SRNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = Type1(1, 64)
            self.l2 = Type1(64, 16)
            self.unpooled = nn.Sequential(*[Type2(16) for _ in range(5)])
            self.pooled = nn.Sequential(
                Type3(16, 16),
                Type3(16, 64),
                Type3(64, 128),
                Type3(128, 256),
            )
            self.l12 = Type4(256, 512)
            self.fc = nn.Linear(512, int(num_classes), bias=False)
            self.reset_parameters()

        def reset_parameters(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)

        def forward(self, x):
            x = self.l1(x)
            x = self.l2(x)
            x = self.unpooled(x)
            x = self.pooled(x)
            x = self.l12(x)
            return self.fc(x)

    return SRNet()


def parameter_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _dihedral_pair(cover: np.ndarray, stego: np.ndarray, code: int):
    k = int(code) % 4
    c = np.rot90(cover, k)
    s = np.rot90(stego, k)
    if int(code) >= 4:
        c = np.fliplr(c)
        s = np.fliplr(s)
    return np.ascontiguousarray(c), np.ascontiguousarray(s)


def score_srnet(model, images, *, device: str | None = None, batch_size: int = 4):
    torch, _, F = _torch()
    if device is None:
        device = next(model.parameters()).device.type
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(images), int(batch_size)):
            arr = np.stack(images[start:start + int(batch_size)]).astype(np.float32)
            x = torch.from_numpy(arr[:, None]).to(device)
            p = F.softmax(model(x), dim=1)[:, 1]
            out.extend(p.detach().cpu().numpy().tolist())
    return np.asarray(out, dtype=float)


def _metric_pair(cover_scores, stego_scores, fixed_fpr: float = 0.05):
    from sklearn.metrics import roc_auc_score, roc_curve

    c = np.asarray(cover_scores, dtype=float)
    s = np.asarray(stego_scores, dtype=float)
    if len(c) != len(s) or len(c) == 0:
        raise ValueError("cover/stego scores must be non-empty and aligned")
    y = np.tile([0, 1], len(c))
    scores = np.column_stack([c, s]).reshape(-1)
    auc = float(roc_auc_score(y, scores))
    fpr, tpr, _ = roc_curve(y, scores)
    idx = np.where(fpr <= float(fixed_fpr))[0]
    tpr_fixed = float(tpr[idx[-1]]) if len(idx) else 0.0
    pe = float(np.min(0.5 * (fpr + (1.0 - tpr))))
    return {"auc": auc, "tpr_at_fpr": tpr_fixed, "pe": pe}


def train_srnet(
    covers,
    stegos,
    *,
    validation=None,
    epochs_stage1: int = 10,
    epochs_stage2: int = 3,
    micro_pair_batch_size: int = 2,
    accumulation_steps: int = 4,
    lr_stage1: float = 1e-3,
    lr_stage2: float = 1e-4,
    weight_decay: float = 2e-4,
    seed: int = 20260918,
    device: str | None = None,
    fixed_fpr: float = 0.05,
):
    """Train SRNet with a fixed two-stage Adamax schedule.

    Cover/stego pairs receive identical dihedral augmentation. Validation is
    diagnostic only: there is no early stopping and no best-epoch selection.
    """
    if len(covers) != len(stegos) or len(covers) == 0:
        raise ValueError("covers/stegos must be non-empty and aligned")
    if int(accumulation_steps) < 1 or int(micro_pair_batch_size) < 1:
        raise ValueError("batch and accumulation parameters must be positive")

    torch, nn, _ = _torch()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_srnet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    decay_params = []
    decay_ids = set()
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            decay_params.append(module.weight)
            decay_ids.add(id(module.weight))
    no_decay_params = [p for p in model.parameters() if id(p) not in decay_ids]
    optimizer = torch.optim.Adamax(
        [
            {"params": decay_params, "weight_decay": float(weight_decay)},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=float(lr_stage1),
    )
    rng = np.random.default_rng(int(seed))
    pair_ids = np.arange(len(covers), dtype=int)
    history = []
    total_epochs = int(epochs_stage1) + int(epochs_stage2)

    for epoch in range(total_epochs):
        lr = float(lr_stage1 if epoch < int(epochs_stage1) else lr_stage2)
        for group in optimizer.param_groups:
            group["lr"] = lr

        rng.shuffle(pair_ids)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        pending = 0

        for start in range(0, len(pair_ids), int(micro_pair_batch_size)):
            ids = pair_ids[start:start + int(micro_pair_batch_size)]
            xs, ys = [], []
            for i in ids:
                code = int(rng.integers(0, 8))
                c, s = _dihedral_pair(covers[int(i)], stegos[int(i)], code)
                xs.extend([c, s])
                ys.extend([0, 1])

            arr = np.stack(xs).astype(np.float32)
            x = torch.from_numpy(arr[:, None]).to(device)
            y = torch.tensor(ys, dtype=torch.long, device=device)
            loss = loss_fn(model(x), y) / int(accumulation_steps)
            loss.backward()
            losses.append(float(loss.detach().cpu()) * int(accumulation_steps))
            pending += 1

            if pending == int(accumulation_steps):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0

        if pending:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        rec = {"epoch": epoch + 1, "lr": lr, "train_loss": float(np.mean(losses))}
        if validation is not None:
            vc, vs = validation
            cs = score_srnet(model, vc, device=device, batch_size=4)
            ss = score_srnet(model, vs, device=device, batch_size=4)
            met = _metric_pair(cs, ss, fixed_fpr=fixed_fpr)
            rec.update({
                "dev_auc": met["auc"],
                "dev_tpr_at_5pct_fpr": met["tpr_at_fpr"],
                "dev_pe": met["pe"],
                "dev_cover_score_mean": float(np.mean(cs)),
                "dev_stego_score_mean": float(np.mean(ss)),
            })
        history.append(rec)

    return model, history, device


def save_srnet(model, path: str | Path, metadata: dict | None = None):
    torch, _, _ = _torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    if metadata is not None:
        path.with_suffix(path.suffix + ".json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )


def load_srnet(path: str | Path, device: str | None = None):
    torch, _, _ = _torch()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_srnet().to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, device


def minimal_detection_error(cover_scores, stego_scores) -> float:
    return _metric_pair(cover_scores, stego_scores, fixed_fpr=0.05)["pe"]


def paired_pe_bootstrap(
    cover_scores,
    method_scores,
    reference_scores,
    *,
    n_resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 20260918,
):
    c = np.asarray(cover_scores, dtype=float)
    a = np.asarray(method_scores, dtype=float)
    b = np.asarray(reference_scores, dtype=float)
    if not (len(c) == len(a) == len(b)) or len(c) == 0:
        raise ValueError("score arrays must be non-empty and aligned")
    rng = np.random.default_rng(int(seed))
    d = np.empty(int(n_resamples), dtype=float)
    n = len(c)
    for k in range(int(n_resamples)):
        idx = rng.integers(0, n, size=n)
        d[k] = (
            minimal_detection_error(c[idx], a[idx])
            - minimal_detection_error(c[idx], b[idx])
        )
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "pe_diff": float(minimal_detection_error(c, a) - minimal_detection_error(c, b)),
        "pe_diff_low": float(np.quantile(d, alpha)),
        "pe_diff_high": float(np.quantile(d, 1.0 - alpha)),
    }
