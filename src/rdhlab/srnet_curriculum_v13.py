from __future__ import annotations

from pathlib import Path
import copy
import json
import os
import random
import time

import numpy as np

from rdhlab.srnet_secondary_v11 import build_srnet, parameter_count, score_srnet


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def _dihedral_pair(cover: np.ndarray, stego: np.ndarray, code: int):
    k = int(code) % 4
    c = np.rot90(cover, k)
    s = np.rot90(stego, k)
    if int(code) >= 4:
        c = np.fliplr(c)
        s = np.fliplr(s)
    return np.ascontiguousarray(c), np.ascontiguousarray(s)


def _configure_model_memory_format(model, device: str):
    torch, _ = _torch()
    if str(device).startswith("cpu"):
        model = model.to(memory_format=torch.channels_last)
    return model


def _to_tensor(arr: np.ndarray, device: str):
    torch, _ = _torch()
    x = torch.from_numpy(arr[:, None]).to(device)
    if str(device).startswith("cpu"):
        x = x.contiguous(memory_format=torch.channels_last)
    return x


def _optimizer_for(model, lr: float, weight_decay: float):
    torch, nn = _torch()
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


def _save_checkpoint(path: Path, *, model, optimizer, state: dict):
    torch, _ = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "state": state,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _load_checkpoint(path: Path, *, model, device: str):
    torch, _ = _torch()
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    return payload


def train_progressive_srnet_v13(
    *,
    stage_specs,
    training_by_payload,
    validation_by_payload,
    pair_batch_size: int = 6,
    weight_decay: float = 2e-4,
    seed: int = 20260919,
    device: str | None = None,
    fixed_fpr: float = 0.05,
    checkpoint_dir: str | Path,
    protocol_sha256: str,
    resume: bool = True,
    log_interval: int = 100,
):
    """Train an SRNet-architecture detector by TRAIN/DEV-only progressive curriculum.

    Each stage is trained and evaluated at one payload. The best DEV-AUC snapshot
    from the current payload seeds the next lower-payload stage. No TEST data are
    accepted by this function.

    Expected stage spec keys:
      name, payload_bpp, min_epochs, max_epochs, lr, pairs_per_epoch
    Optional keys:
      early_success_auc, fail_check_epoch, fail_below_auc
    """
    if int(pair_batch_size) < 1:
        raise ValueError("pair_batch_size must be positive")
    if not stage_specs:
        raise ValueError("stage_specs must be non-empty")

    torch, nn = _torch()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_srnet().to(device)
    model = _configure_model_memory_format(model, device)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(int(seed))

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "srnet_v13_last.pt"

    history = []
    stage_summaries = []
    start_stage = 0
    start_epoch_in_stage = 0
    best_stage_auc = -np.inf
    best_stage_epoch = None
    optimizer = None

    if resume and last_path.exists():
        payload = _load_checkpoint(last_path, model=model, device=device)
        state = payload.get("state", {})
        if state.get("protocol_sha256") != protocol_sha256:
            raise RuntimeError("Existing SRNet-v13 checkpoint belongs to a different protocol.")
        history = list(state.get("history", []))
        stage_summaries = list(state.get("stage_summaries", []))
        start_stage = int(state.get("stage_index", 0))
        start_epoch_in_stage = int(state.get("epoch_in_stage", 0))
        best_stage_auc = float(state.get("best_stage_auc", -np.inf))
        best_stage_epoch = state.get("best_stage_epoch")
        if "rng_state" in state:
            rng.bit_generator.state = state["rng_state"]
        if bool(state.get("halted", False)):
            raise RuntimeError(
                "Existing SRNet-v13 checkpoint is marked halted by a failed progress gate. "
                "Preserve it; start a new protocol rather than silently continuing the same run."
            )
        if start_stage < len(stage_specs) and start_epoch_in_stage > 0:
            optimizer = _optimizer_for(model, stage_specs[start_stage]["lr"], weight_decay)
            if payload.get("optimizer_state") is not None:
                optimizer.load_state_dict(payload["optimizer_state"])
        print(
            f"RESUME: stage {start_stage + 1}/{len(stage_specs)}, "
            f"epoch_in_stage={start_epoch_in_stage}",
            flush=True,
        )

    for stage_idx in range(start_stage, len(stage_specs)):
        spec = dict(stage_specs[stage_idx])
        name = str(spec["name"])
        p = float(spec["payload_bpp"])
        min_epochs = int(spec["min_epochs"])
        max_epochs = int(spec["max_epochs"])
        lr = float(spec["lr"])
        pairs_per_epoch = int(spec["pairs_per_epoch"])

        if p not in training_by_payload or p not in validation_by_payload:
            raise KeyError(f"Missing training/validation data for payload {p}")
        covers, stegos = training_by_payload[p]
        val_c, val_s = validation_by_payload[p]
        if len(covers) != len(stegos) or len(covers) == 0:
            raise ValueError(f"Invalid training pair arrays for payload {p}")
        if len(val_c) != len(val_s) or len(val_c) == 0:
            raise ValueError(f"Invalid development pair arrays for payload {p}")

        stage_best_path = checkpoint_dir / f"stage_{stage_idx + 1:02d}_{name}_best.pt"

        epoch0 = start_epoch_in_stage if stage_idx == start_stage else 0
        if epoch0 == 0:
            optimizer = _optimizer_for(model, lr, weight_decay)
            best_stage_auc = -np.inf
            best_stage_epoch = None
        elif optimizer is None:
            optimizer = _optimizer_for(model, lr, weight_decay)

        stage_status = "MAX_EPOCHS_REACHED"
        for epoch_in_stage in range(epoch0, max_epochs):
            for group in optimizer.param_groups:
                group["lr"] = lr

            n_available = len(covers)
            n_epoch = min(pairs_per_epoch, n_available)
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
                        f"stage {stage_idx + 1:02d}/{len(stage_specs)} | {name} | "
                        f"epoch {epoch_in_stage + 1:02d}/{max_epochs:02d} | "
                        f"batch {b:04d}/{num_batches:04d} | "
                        f"loss={np.mean(losses[-min(len(losses),100):]):.4f} | "
                        f"elapsed={elapsed:.1f} min",
                        flush=True,
                    )

            cs = score_srnet(model, val_c, device=device, batch_size=max(4, int(pair_batch_size)))
            ss = score_srnet(model, val_s, device=device, batch_size=max(4, int(pair_batch_size)))
            met = _metric_pair(cs, ss, fixed_fpr=fixed_fpr)
            elapsed = (time.perf_counter() - t0) / 60.0

            global_epoch = len(history) + 1
            rec = {
                "global_epoch": int(global_epoch),
                "stage_index": int(stage_idx + 1),
                "stage": name,
                "payload_bpp": p,
                "epoch_in_stage": int(epoch_in_stage + 1),
                "lr": lr,
                "pairs_sampled": int(n_epoch),
                "pair_batch_size": int(pair_batch_size),
                "train_loss": float(np.mean(losses)),
                "dev_pairs": int(len(val_c)),
                "dev_auc": float(met["auc"]),
                "dev_tpr_at_5pct_fpr": float(met["tpr_at_fpr"]),
                "dev_pe": float(met["pe"]),
                "dev_cover_score_mean": float(np.mean(cs)),
                "dev_stego_score_mean": float(np.mean(ss)),
                "epoch_minutes": float(elapsed),
            }
            history.append(rec)

            if met["auc"] > best_stage_auc:
                best_stage_auc = float(met["auc"])
                best_stage_epoch = int(epoch_in_stage + 1)
                _save_checkpoint(
                    stage_best_path,
                    model=model,
                    optimizer=optimizer,
                    state={
                        "protocol_sha256": protocol_sha256,
                        "stage_index": int(stage_idx),
                        "stage_name": name,
                        "payload_bpp": p,
                        "selected_epoch_in_stage": best_stage_epoch,
                        "dev_auc": best_stage_auc,
                    },
                )

            next_epoch = int(epoch_in_stage + 1)
            state = {
                "protocol_sha256": protocol_sha256,
                "stage_index": int(stage_idx),
                "epoch_in_stage": next_epoch,
                "history": history,
                "stage_summaries": stage_summaries,
                "best_stage_auc": float(best_stage_auc),
                "best_stage_epoch": best_stage_epoch,
                "rng_state": copy.deepcopy(rng.bit_generator.state),
            }
            _save_checkpoint(last_path, model=model, optimizer=optimizer, state=state)

            print(
                f"EPOCH COMPLETE: stage={name}, payload={p:.3f}, "
                f"epoch={epoch_in_stage + 1}/{max_epochs}, "
                f"train_loss={rec['train_loss']:.4f}, dev_AUC={rec['dev_auc']:.4f}, "
                f"dev_Pe={rec['dev_pe']:.4f}, best_stage_AUC={best_stage_auc:.4f}, "
                f"time={elapsed:.1f} min",
                flush=True,
            )

            fail_check_epoch = spec.get("fail_check_epoch")
            fail_below_auc = spec.get("fail_below_auc")
            if (
                fail_check_epoch is not None
                and fail_below_auc is not None
                and (epoch_in_stage + 1) >= int(fail_check_epoch)
                and best_stage_auc < float(fail_below_auc)
            ):
                stage_status = "FAILED_PROGRESS_GATE"
                break

            success_auc = spec.get("early_success_auc")
            if (
                success_auc is not None
                and (epoch_in_stage + 1) >= min_epochs
                and best_stage_auc >= float(success_auc)
            ):
                stage_status = "EARLY_SUCCESS"
                break

        if not stage_best_path.exists():
            raise RuntimeError(f"No checkpoint produced for stage {name}")

        _load_checkpoint(stage_best_path, model=model, device=device)
        model = _configure_model_memory_format(model, device)
        summary = {
            "stage_index": int(stage_idx + 1),
            "stage": name,
            "payload_bpp": p,
            "status": stage_status,
            "best_dev_auc": float(best_stage_auc),
            "best_epoch_in_stage": int(best_stage_epoch),
            "best_checkpoint": str(stage_best_path),
        }
        stage_summaries.append(summary)

        if stage_status == "FAILED_PROGRESS_GATE":
            final_state = {
                "protocol_sha256": protocol_sha256,
                "stage_index": int(stage_idx),
                "epoch_in_stage": int(max_epochs),
                "history": history,
                "stage_summaries": stage_summaries,
                "best_stage_auc": float(best_stage_auc),
                "best_stage_epoch": best_stage_epoch,
                "rng_state": copy.deepcopy(rng.bit_generator.state),
                "halted": True,
                "halt_reason": "FAILED_PROGRESS_GATE",
            }
            _save_checkpoint(last_path, model=model, optimizer=None, state=final_state)
            return model, history, stage_summaries, device, {
                "status": "FAILED_PROGRESS_GATE",
                "failed_stage": name,
                "failed_payload_bpp": p,
                "best_dev_auc": float(best_stage_auc),
            }

        # Save a transition checkpoint containing the selected best snapshot.
        transition_state = {
            "protocol_sha256": protocol_sha256,
            "stage_index": int(stage_idx + 1),
            "epoch_in_stage": 0,
            "history": history,
            "stage_summaries": stage_summaries,
            "best_stage_auc": -np.inf,
            "best_stage_epoch": None,
            "rng_state": copy.deepcopy(rng.bit_generator.state),
            "halted": False,
        }
        _save_checkpoint(last_path, model=model, optimizer=None, state=transition_state)
        start_epoch_in_stage = 0

    return model, history, stage_summaries, device, {
        "status": "COMPLETE",
        "final_stage": str(stage_specs[-1]["name"]),
        "final_payload_bpp": float(stage_specs[-1]["payload_bpp"]),
        "selected_dev_auc": float(stage_summaries[-1]["best_dev_auc"]),
        "selected_epoch_in_stage": int(stage_summaries[-1]["best_epoch_in_stage"]),
        "selected_checkpoint": str(stage_summaries[-1]["best_checkpoint"]),
    }


def save_state_dict(model, path: str | Path):
    torch, _ = _torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_state_dict(path: str | Path, device: str | None = None):
    torch, _ = _torch()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_srnet().to(device)
    model = _configure_model_memory_format(model, device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model, device
