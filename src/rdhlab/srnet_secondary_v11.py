from __future__ import annotations

from pathlib import Path
import copy
import json
import os
import random
import time

import numpy as np


def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def build_srnet(num_classes: int = 2):
    """Build SRNet for grayscale steganalysis (Boroumand et al., TIFS 2019).

    The topology follows the published network: two Type-1 layers, five
    unpooled residual Type-2 layers, four pooled residual Type-3 layers,
    one Type-4 layer with global average pooling, and a bias-free linear
    classifier. Convolutional kernels are learned end-to-end.
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


def _configure_model_memory_format(model, device: str):
    torch, _, _ = _torch()
    if str(device).startswith("cpu"):
        # oneDNN/CPU convolution is often more efficient in channels-last.
        model = model.to(memory_format=torch.channels_last)
    return model


def _to_tensor(arr: np.ndarray, device: str):
    torch, _, _ = _torch()
    x = torch.from_numpy(arr[:, None]).to(device)
    if str(device).startswith("cpu"):
        x = x.contiguous(memory_format=torch.channels_last)
    return x


def score_srnet(model, images, *, device: str | None = None, batch_size: int = 8):
    torch, _, F = _torch()
    if device is None:
        device = next(model.parameters()).device.type
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(images), int(batch_size)):
            arr = np.stack(images[start:start + int(batch_size)]).astype(np.float32, copy=False)
            x = _to_tensor(arr, device)
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


def _optimizer_for(model, lr: float, weight_decay: float):
    torch, nn, _ = _torch()
    decay_params = []
    decay_ids = set()
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            decay_params.append(module.weight)
            decay_ids.add(id(module.weight))
    no_decay_params = [p for p in model.parameters() if id(p) not in decay_ids]
    return torch.optim.Adamax(
        [
            {"params": decay_params, "weight_decay": float(weight_decay)},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=float(lr),
    )


def _save_training_checkpoint(path: Path, *, model, optimizer, state: dict):
    torch, _, _ = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "state": state,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train_srnet_v11(
    *,
    curriculum_covers,
    curriculum_stegos,
    target_covers,
    target_stegos,
    validation,
    curriculum_epochs: int = 2,
    target_epochs_stage1: int = 7,
    target_epochs_stage2: int = 4,
    pair_batch_size: int = 4,
    pairs_per_epoch: int = 3000,
    lr_stage1: float = 1e-3,
    lr_stage2: float = 1e-4,
    weight_decay: float = 2e-4,
    seed: int = 20260919,
    device: str | None = None,
    fixed_fpr: float = 0.05,
    checkpoint_dir: str | Path | None = None,
    protocol_sha256: str | None = None,
    resume: bool = True,
    log_interval: int = 100,
):
    """CPU-oriented SRNet detector-development run with curriculum and resume.

    Scientific constraints:
    - curriculum and target data are TRAIN-only;
    - validation is TRAIN-development only;
    - checkpoint selection uses target-payload validation AUC only;
    - TEST is never accessed here;
    - the best checkpoint is selected only during the low-learning-rate
      target stage, mirroring the published SRNet validation-snapshot logic.

    The CPU schedule is deliberately much shorter than the published 500k
    iteration schedule and uses a smaller real minibatch. It must therefore be
    described as an SRNet-architecture detector rather than a reproduction of
    the original detector's training regime.
    """
    if len(curriculum_covers) != len(curriculum_stegos) or len(curriculum_covers) == 0:
        raise ValueError("curriculum cover/stego arrays must be non-empty and aligned")
    if len(target_covers) != len(target_stegos) or len(target_covers) == 0:
        raise ValueError("target cover/stego arrays must be non-empty and aligned")
    if int(pair_batch_size) < 1 or int(pairs_per_epoch) < 1:
        raise ValueError("pair_batch_size and pairs_per_epoch must be positive")

    torch, nn, _ = _torch()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_srnet().to(device)
    model = _configure_model_memory_format(model, device)
    optimizer = _optimizer_for(model, lr_stage1, weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(int(seed))

    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    last_path = checkpoint_dir / "srnet_v11_last.pt" if checkpoint_dir else None
    best_path = checkpoint_dir / "srnet_v11_best.pt" if checkpoint_dir else None

    stages = (
        [("curriculum_0.012", float(lr_stage1), curriculum_covers, curriculum_stegos)] * int(curriculum_epochs)
        + [("target_0.009_high_lr", float(lr_stage1), target_covers, target_stegos)] * int(target_epochs_stage1)
        + [("target_0.009_low_lr", float(lr_stage2), target_covers, target_stegos)] * int(target_epochs_stage2)
    )

    history = []
    next_epoch = 0
    best_auc = -np.inf
    best_epoch = None

    if resume and last_path is not None and last_path.exists():
        payload = torch.load(last_path, map_location=device, weights_only=False)
        state = payload.get("state", {})
        if protocol_sha256 is not None and state.get("protocol_sha256") != protocol_sha256:
            raise RuntimeError("Existing SRNet-v11 checkpoint belongs to a different pre-test protocol.")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        next_epoch = int(state.get("next_epoch", 0))
        history = list(state.get("history", []))
        best_auc = float(state.get("best_auc", -np.inf))
        best_epoch = state.get("best_epoch")
        if "rng_state" in state:
            rng.bit_generator.state = state["rng_state"]
        print(f"RESUME: epoch {next_epoch}/{len(stages)}, best dev AUC={best_auc:.4f}")

    val_c, val_s = validation

    for epoch_idx in range(next_epoch, len(stages)):
        stage_name, lr, covers, stegos = stages[epoch_idx]
        for group in optimizer.param_groups:
            group["lr"] = float(lr)

        n_available = len(covers)
        n_epoch = min(int(pairs_per_epoch), n_available)
        chosen = rng.choice(n_available, size=n_epoch, replace=False)
        rng.shuffle(chosen)

        model.train()
        losses = []
        t0 = time.perf_counter()
        num_batches = (n_epoch + int(pair_batch_size) - 1) // int(pair_batch_size)

        for b, start in enumerate(range(0, n_epoch, int(pair_batch_size)), 1):
            ids = chosen[start:start + int(pair_batch_size)]
            xs, ys = [], []
            for i in ids:
                code = int(rng.integers(0, 8))
                c, s = _dihedral_pair(covers[int(i)], stegos[int(i)], code)
                xs.extend([c, s])
                ys.extend([0, 1])

            arr = np.stack(xs).astype(np.float32, copy=False)
            x = _to_tensor(arr, device)
            y = torch.tensor(ys, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

            if int(log_interval) > 0 and (b % int(log_interval) == 0 or b == num_batches):
                elapsed = (time.perf_counter() - t0) / 60.0
                print(
                    f"epoch {epoch_idx + 1:02d}/{len(stages)} | {stage_name} | "
                    f"batch {b:04d}/{num_batches} | loss={np.mean(losses[-min(len(losses),100):]):.4f} | "
                    f"elapsed={elapsed:.1f} min",
                    flush=True,
                )

        cs = score_srnet(model, val_c, device=device, batch_size=max(4, int(pair_batch_size)))
        ss = score_srnet(model, val_s, device=device, batch_size=max(4, int(pair_batch_size)))
        met = _metric_pair(cs, ss, fixed_fpr=fixed_fpr)
        elapsed = (time.perf_counter() - t0) / 60.0
        rec = {
            "epoch": epoch_idx + 1,
            "stage": stage_name,
            "lr": float(lr),
            "pairs_sampled": int(n_epoch),
            "pair_batch_size": int(pair_batch_size),
            "train_loss": float(np.mean(losses)),
            "dev_auc": met["auc"],
            "dev_tpr_at_5pct_fpr": met["tpr_at_fpr"],
            "dev_pe": met["pe"],
            "dev_cover_score_mean": float(np.mean(cs)),
            "dev_stego_score_mean": float(np.mean(ss)),
            "epoch_minutes": float(elapsed),
        }
        history.append(rec)

        is_selection_stage = stage_name == "target_0.009_low_lr"
        if is_selection_stage and met["auc"] > best_auc:
            best_auc = float(met["auc"])
            best_epoch = int(epoch_idx + 1)
            if best_path is not None:
                _save_training_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    state={
                        "protocol_sha256": protocol_sha256,
                        "selected_epoch": best_epoch,
                        "selected_stage": stage_name,
                        "dev_auc": best_auc,
                    },
                )

        state = {
            "protocol_sha256": protocol_sha256,
            "next_epoch": int(epoch_idx + 1),
            "history": history,
            "best_auc": float(best_auc),
            "best_epoch": best_epoch,
            "rng_state": copy.deepcopy(rng.bit_generator.state),
            "completed": bool(epoch_idx + 1 == len(stages)),
        }
        if last_path is not None:
            _save_training_checkpoint(last_path, model=model, optimizer=optimizer, state=state)

        print(
            f"EPOCH COMPLETE {epoch_idx + 1:02d}/{len(stages)}: "
            f"stage={stage_name}, train_loss={rec['train_loss']:.4f}, "
            f"dev_AUC={rec['dev_auc']:.4f}, dev_Pe={rec['dev_pe']:.4f}, "
            f"time={elapsed:.1f} min, best_lowLR_AUC={best_auc:.4f}",
            flush=True,
        )

    if best_path is None or not best_path.exists():
        raise RuntimeError("No selectable low-learning-rate target checkpoint was produced.")

    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state"])
    model.eval()
    return model, history, device, {
        "best_auc": float(best_auc),
        "best_epoch": int(best_epoch),
        "best_checkpoint": str(best_path),
    }


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
    model = _configure_model_memory_format(model, device)
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
    seed: int = 20260919,
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
