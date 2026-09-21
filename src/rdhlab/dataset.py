from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

LOSSLESS_IMAGE_EXTENSIONS = {".pgm", ".png", ".bmp", ".tif", ".tiff"}
LOSSY_IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
SUPPORTED_IMAGE_EXTENSIONS = LOSSLESS_IMAGE_EXTENSIONS | LOSSY_IMAGE_EXTENSIONS
DEFAULT_SEED = 20260916


@dataclass(frozen=True)
class ImageCollectionCandidate:
    relative_dir: str
    image_count: int
    extensions: str
    lossy: bool
    quality_factor: int | None


@dataclass(frozen=True)
class BossbasePreparationResult:
    archive_path: str
    archive_sha256: str
    extracted_dir: str
    source_dir: str
    source_label: str
    source_is_lossy: bool
    prepared_dir: str
    manifest_path: str
    metadata_path: str
    image_count: int
    train_count: int
    validation_count: int
    test_count: int
    converted_count: int
    reused_count: int


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(zip_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"Unsafe archive member path: {member.filename}")
        archive.extractall(destination)


def _extract_nested_zip_archives(root: Path, *, max_depth: int = 4) -> list[Path]:
    extracted: list[Path] = []
    seen: set[Path] = set()
    for _depth in range(max_depth):
        archives = [z for z in root.rglob("*.zip") if z.is_file() and z.resolve() not in seen]
        if not archives:
            break
        progress = False
        for archive in sorted(archives):
            resolved = archive.resolve()
            seen.add(resolved)
            destination = archive.parent / f"{archive.stem}_extracted"
            if destination.exists():
                shutil.rmtree(destination)
            destination.mkdir(parents=True, exist_ok=True)
            _safe_extract(archive, destination)
            extracted.append(destination)
            progress = True
        if not progress:
            break
    return extracted


def _archive_diagnostics(root: Path, *, limit: int = 25) -> str:
    files = [p for p in root.rglob("*") if p.is_file()]
    suffix_counts: dict[str, int] = {}
    for p in files:
        suffix = p.suffix.lower() or "<no extension>"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    suffix_summary = ", ".join(
        f"{suffix}:{count}"
        for suffix, count in sorted(suffix_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
    ) or "<no files>"
    sample = "\n".join(f"  - {p.relative_to(root).as_posix()}" for p in files[:limit])
    nested = [p.relative_to(root).as_posix() for p in files if p.suffix.lower() == ".zip"]
    nested_text = ", ".join(nested[:10]) if nested else "none"
    return (
        f"Extracted file-type counts: {suffix_summary}.\n"
        f"Nested ZIP files still present: {nested_text}.\n"
        f"First extracted files:\n{sample if sample else '  <none>'}"
    )


def find_bossbase_archive(raw_dir: str | Path) -> Path:
    raw_dir = Path(raw_dir)
    archives = sorted(raw_dir.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"No ZIP archive found in {raw_dir}. Copy the dataset archive there first.")

    preferred = [p for p in archives if "boss" in p.name.lower()]
    if len(preferred) == 1:
        return preferred[0]
    if len(preferred) > 1:
        names = ", ".join(p.name for p in preferred)
        raise RuntimeError(f"Multiple BOSSbase-like ZIP archives found: {names}")
    if len(archives) == 1:
        return archives[0]

    names = ", ".join(p.name for p in archives)
    raise RuntimeError(
        "Multiple ZIP archives found and none has 'boss' in its filename. "
        f"Rename the intended archive to BOSSbase_1.01.zip. Found: {names}"
    )


def _numeric_sort_key(path: Path) -> tuple[int, int | str, str]:
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem), path.as_posix())
    return (1, stem.lower(), path.as_posix())


def discover_images(root: str | Path, *, include_lossy: bool = False) -> list[Path]:
    root = Path(root)
    extensions = SUPPORTED_IMAGE_EXTENSIONS if include_lossy else LOSSLESS_IMAGE_EXTENSIONS
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions]
    return sorted(files, key=_numeric_sort_key)


def _qf_from_path(path: Path) -> int | None:
    for part in reversed(path.parts):
        match = re.fullmatch(r"QF[_-]?(\d{1,3})", part, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def discover_image_collections(root: str | Path, *, expected_count: int = 10_000) -> list[ImageCollectionCandidate]:
    """Find leaf-like directories that directly contain one complete image collection.

    This is intentionally directory-based: a bundle containing QF75/QF90/QF95
    should be reported as three 10k candidates, not one misleading 30k dataset.
    """
    root = Path(root)
    groups: dict[Path, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            groups.setdefault(path.parent, []).append(path)

    candidates: list[ImageCollectionCandidate] = []
    for directory, files in sorted(groups.items(), key=lambda kv: kv[0].as_posix()):
        if len(files) != expected_count:
            continue
        ext_counts: dict[str, int] = {}
        for p in files:
            ext = p.suffix.lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        lossy = any(ext in LOSSY_IMAGE_EXTENSIONS for ext in ext_counts)
        extensions = ",".join(f"{k}:{v}" for k, v in sorted(ext_counts.items()))
        candidates.append(
            ImageCollectionCandidate(
                relative_dir=directory.relative_to(root).as_posix() or ".",
                image_count=len(files),
                extensions=extensions,
                lossy=lossy,
                quality_factor=_qf_from_path(directory),
            )
        )
    return candidates



def discover_archive_image_collections(
    archive_path: str | Path, *, expected_count: int = 10_000
) -> list[ImageCollectionCandidate]:
    """Inspect image collections from the ZIP central directory without extracting files.

    This avoids tens of thousands of filesystem ``stat`` calls on Docker Desktop bind mounts,
    which can be fragile on secondary Windows drives.
    """
    archive_path = Path(archive_path)
    groups: dict[str, list[str]] = {}
    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            member = info.filename.replace("\\", "/")
            suffix = Path(member).suffix.lower()
            if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            parent = Path(member).parent.as_posix()
            groups.setdefault(parent, []).append(member)

    candidates: list[ImageCollectionCandidate] = []
    for directory, members in sorted(groups.items()):
        if len(members) != expected_count:
            continue
        ext_counts: dict[str, int] = {}
        for member in members:
            ext = Path(member).suffix.lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        lossy = any(ext in LOSSY_IMAGE_EXTENSIONS for ext in ext_counts)
        extensions = ",".join(f"{k}:{v}" for k, v in sorted(ext_counts.items()))
        candidates.append(
            ImageCollectionCandidate(
                relative_dir=directory or ".",
                image_count=len(members),
                extensions=extensions,
                lossy=lossy,
                quality_factor=_qf_from_path(Path(directory)),
            )
        )
    return candidates


def _archive_members_in_collection(archive_path: Path, relative_dir: str) -> list[str]:
    prefix = "" if relative_dir in ("", ".") else relative_dir.rstrip("/") + "/"
    members: list[str] = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            member = info.filename.replace("\\", "/")
            if Path(member).suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            parent = Path(member).parent.as_posix()
            if parent == relative_dir or (relative_dir == "." and parent == "."):
                members.append(member)
    return sorted(members, key=lambda m: _numeric_sort_key(Path(m)))


def _select_archive_collection(
    archive_path: Path,
    *,
    expected_count: int,
    source_subdir: str | None,
    allow_lossy_source: bool,
) -> tuple[list[str], str, bool] | None:
    """Select a complete collection directly from a ZIP, or return None for nested archives."""
    candidates = discover_archive_image_collections(archive_path, expected_count=expected_count)

    # Canonical/other single lossless dataset: accept exactly expected_count lossless image
    # members even when they live below one directory.
    with zipfile.ZipFile(archive_path, "r") as zf:
        all_images = [
            info.filename.replace("\\", "/")
            for info in zf.infolist()
            if not info.is_dir() and Path(info.filename).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
    lossless_all = [m for m in all_images if Path(m).suffix.lower() in LOSSLESS_IMAGE_EXTENSIONS]
    if source_subdir is None and len(lossless_all) == expected_count and len(all_images) == expected_count:
        return sorted(lossless_all, key=lambda m: _numeric_sort_key(Path(m))), "BOSSbase 1.01 (lossless source)", False

    if not candidates:
        return None

    if source_subdir is not None:
        if source_subdir.upper() == "AUTO_HIGHEST_QF":
            qf_candidates = [c for c in candidates if c.quality_factor is not None]
            if not qf_candidates:
                raise RuntimeError(
                    "AUTO_HIGHEST_QF was requested, but no QFxx image collection was found in the ZIP.\n"
                    "Candidates:\n" + _format_candidates(candidates)
                )
            chosen = max(qf_candidates, key=lambda c: (c.quality_factor or -1, c.relative_dir))
        else:
            normalized = Path(source_subdir).as_posix().rstrip("/")
            matches = [c for c in candidates if c.relative_dir.rstrip("/") == normalized]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Requested source_subdir={source_subdir!r} was not found as a complete "
                    f"{expected_count}-image collection in the ZIP.\nCandidates:\n"
                    + _format_candidates(candidates)
                )
            chosen = matches[0]
    elif len(candidates) == 1:
        chosen = candidates[0]
    else:
        raise RuntimeError(
            f"Expected one {expected_count}-image collection, but the ZIP is ambiguous.\n"
            "Detected complete collections:\n" + _format_candidates(candidates)
            + "\nChoose one explicitly with source_subdir='grayscale/QF95' (example), or use "
            "source_subdir='AUTO_HIGHEST_QF'."
        )

    if chosen.lossy and not allow_lossy_source:
        raise RuntimeError(
            f"Selected collection {chosen.relative_dir!r} contains JPEG images. "
            "This is a lossy BOSSbase-derived dataset, not canonical lossless BOSSbase 1.01. "
            "Set allow_lossy_source=True only if this distinction is intentional."
        )

    members = _archive_members_in_collection(archive_path, chosen.relative_dir)
    if chosen.lossy:
        qf_text = f", QF{chosen.quality_factor}" if chosen.quality_factor is not None else ""
        label = f"BOSSbase-derived JPEG ({chosen.relative_dir}{qf_text})"
    else:
        label = f"BOSSbase collection ({chosen.relative_dir})"
    return members, label, chosen.lossy


def _read_u8_gray_bytes(data: bytes, label: str, *, strict_grayscale: bool = False) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        if strict_grayscale and image.mode != "L":
            raise ValueError(
                f"Input must decode natively as 8-bit grayscale (mode L), got {image.mode!r}: {label}"
            )
        arr = np.asarray(image if image.mode == "L" else image.convert("L"))
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image: {label}")
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected 8-bit grayscale pixels, got dtype={arr.dtype}: {label}")
    return arr


def _exists_with_retry(path: Path, attempts: int = 4, delay: float = 0.15) -> bool:
    last: OSError | None = None
    for attempt in range(attempts):
        try:
            return path.exists()
        except OSError as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(delay * (attempt + 1))
    assert last is not None
    raise last

def _format_candidates(candidates: list[ImageCollectionCandidate]) -> str:
    if not candidates:
        return "  <none>"
    rows = []
    for c in candidates:
        qf = f", QF={c.quality_factor}" if c.quality_factor is not None else ""
        rows.append(
            f"  - {c.relative_dir}: {c.image_count} images, {c.extensions}, "
            f"{'lossy JPEG source' if c.lossy else 'lossless source'}{qf}"
        )
    return "\n".join(rows)


def _resolve_source_collection(
    extracted_dir: Path,
    *,
    expected_count: int,
    source_subdir: str | None,
    allow_lossy_source: bool,
) -> tuple[Path, list[Path], str, bool]:
    # Canonical case first: exactly expected_count lossless images anywhere in the archive.
    lossless = discover_images(extracted_dir, include_lossy=False)
    if source_subdir is None and len(lossless) == expected_count:
        return extracted_dir, lossless, "BOSSbase 1.01 (lossless source)", False

    candidates = discover_image_collections(extracted_dir, expected_count=expected_count)

    if source_subdir is not None:
        if source_subdir.upper() == "AUTO_HIGHEST_QF":
            qf_candidates = [c for c in candidates if c.quality_factor is not None]
            if not qf_candidates:
                raise RuntimeError(
                    "AUTO_HIGHEST_QF was requested, but no QFxx image collection was found.\n"
                    "Candidates:\n" + _format_candidates(candidates)
                )
            chosen = max(qf_candidates, key=lambda c: (c.quality_factor or -1, c.relative_dir))
        else:
            normalized = Path(source_subdir).as_posix().rstrip("/")
            matches = [c for c in candidates if c.relative_dir.rstrip("/") == normalized]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Requested source_subdir={source_subdir!r} was not found as a complete "
                    f"{expected_count}-image collection.\nCandidates:\n" + _format_candidates(candidates)
                )
            chosen = matches[0]

        if chosen.lossy and not allow_lossy_source:
            raise RuntimeError(
                f"Selected collection {chosen.relative_dir!r} contains JPEG images. "
                "This is a lossy BOSSbase-derived dataset, not the canonical lossless BOSSbase 1.01 archive. "
                "Set allow_lossy_source=True only if this distinction is intentional."
            )
        source_dir = extracted_dir / chosen.relative_dir
        images = sorted(
            [p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS],
            key=_numeric_sort_key,
        )
        if chosen.lossy:
            qf_text = f", QF{chosen.quality_factor}" if chosen.quality_factor is not None else ""
            label = f"BOSSbase-derived JPEG ({chosen.relative_dir}{qf_text})"
        else:
            label = f"BOSSbase collection ({chosen.relative_dir})"
        return source_dir, images, label, chosen.lossy

    if len(candidates) == 1:
        chosen = candidates[0]
        if chosen.lossy and not allow_lossy_source:
            raise RuntimeError(
                "The archive contains one complete JPEG image collection, but JPEG is lossy. "
                "Set allow_lossy_source=True only if you intend to use a BOSSbase-derived JPEG dataset.\n"
                "Candidate:\n" + _format_candidates(candidates)
            )
        source_dir = extracted_dir / chosen.relative_dir
        images = sorted(
            [p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS],
            key=_numeric_sort_key,
        )
        label = (
            f"BOSSbase-derived JPEG ({chosen.relative_dir})"
            if chosen.lossy else f"BOSSbase collection ({chosen.relative_dir})"
        )
        return source_dir, images, label, chosen.lossy

    raise RuntimeError(
        f"Expected one {expected_count}-image BOSSbase collection, but the archive is ambiguous.\n"
        "Detected complete collections:\n"
        + _format_candidates(candidates)
        + "\nChoose one explicitly with source_subdir='grayscale/QF95' (example), or use "
        "source_subdir='AUTO_HIGHEST_QF' to select the highest detected JPEG quality factor."
    )


def deterministic_split(
    n: int,
    seed: int = DEFAULT_SEED,
    train_count: int = 6000,
    validation_count: int = 2000,
    test_count: int = 2000,
) -> list[str]:
    if train_count + validation_count + test_count != n:
        raise ValueError(
            "Split counts must sum to the number of images: "
            f"{train_count}+{validation_count}+{test_count}!={n}"
        )
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    rng.shuffle(order)
    labels = [""] * n
    for rank, index in enumerate(order):
        if rank < train_count:
            labels[int(index)] = "train"
        elif rank < train_count + validation_count:
            labels[int(index)] = "validation"
        else:
            labels[int(index)] = "test"
    return labels


def _read_u8_gray(path: Path, *, strict_grayscale: bool = False) -> np.ndarray:
    with Image.open(path) as image:
        if strict_grayscale and image.mode != "L":
            raise ValueError(
                f"Input must decode natively as 8-bit grayscale (mode L), got {image.mode!r}: {path}"
            )
        arr = np.asarray(image if image.mode == "L" else image.convert("L"))
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image: {path}")
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected 8-bit grayscale pixels, got dtype={arr.dtype}: {path}")
    return arr


def _pixel_sha256(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _write_verified_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(path, format="PNG", optimize=False)
    with Image.open(path) as check:
        roundtrip = np.asarray(check, dtype=np.uint8)
    if not np.array_equal(arr, roundtrip):
        raise RuntimeError(f"Lossless PNG verification failed for {path}")


def prepare_bossbase(
    raw_dir: str | Path = "/workspace/data/raw",
    processed_dir: str | Path = "/workspace/data/processed/BOSSbase_primary",
    archive_path: str | Path | None = None,
    expected_count: int = 10_000,
    expected_shape: tuple[int, int] | None = (512, 512),
    seed: int = DEFAULT_SEED,
    force_extract: bool = False,
    source_subdir: str | None = None,
    allow_lossy_source: bool = False,
) -> BossbasePreparationResult:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    archive = Path(archive_path) if archive_path is not None else find_bossbase_archive(raw_dir)
    if not archive.exists():
        raise FileNotFoundError(archive)

    archive_sha = sha256_file(archive)
    extracted_dir = raw_dir / "BOSSbase_1.01_extracted"

    # v0.1.5: prefer streaming image members directly from the ZIP.  This avoids
    # 30k+ Path.stat()/is_file() calls against Docker Desktop's Windows bind mount,
    # the exact operation that can raise OSError [Errno 14] Bad address on secondary drives.
    direct_selection = _select_archive_collection(
        archive,
        expected_count=expected_count,
        source_subdir=source_subdir,
        allow_lossy_source=allow_lossy_source,
    )

    archive_mode = direct_selection is not None
    if archive_mode:
        members, source_label, source_is_lossy = direct_selection
        if len(members) != expected_count:
            raise RuntimeError(
                f"Internal ZIP selection error: expected {expected_count}, selected {len(members)}."
            )
        source_collection = Path(members[0]).parent.as_posix() if members else "."
        source_dir_text = f"zip://{archive.as_posix()}!/{source_collection}"
        print(
            f"Using direct ZIP streaming for {len(members)} images from {source_collection}. "
            "The previously extracted raw directory is ignored."
        )
        images: list[Path] = []
    else:
        marker = extracted_dir / ".extracted_from_sha256"
        need_extract = force_extract or not extracted_dir.exists()
        if extracted_dir.exists() and marker.exists():
            recorded = marker.read_text(encoding="utf-8").strip()
            if recorded != archive_sha:
                need_extract = True
        elif extracted_dir.exists() and not marker.exists():
            need_extract = True

        if need_extract:
            if extracted_dir.exists():
                shutil.rmtree(extracted_dir)
            extracted_dir.mkdir(parents=True, exist_ok=True)
            _safe_extract(archive, extracted_dir)
            marker.write_text(archive_sha + "\n", encoding="utf-8")

        lossless = discover_images(extracted_dir, include_lossy=False)
        candidates = discover_image_collections(extracted_dir, expected_count=expected_count)
        nested_extracted: list[Path] = []
        if len(lossless) != expected_count and not candidates:
            nested_extracted = _extract_nested_zip_archives(extracted_dir)

        try:
            source_dir, images, source_label, source_is_lossy = _resolve_source_collection(
                extracted_dir,
                expected_count=expected_count,
                source_subdir=source_subdir,
                allow_lossy_source=allow_lossy_source,
            )
        except RuntimeError as exc:
            diagnostics = _archive_diagnostics(extracted_dir)
            nested_note = (
                f"\nRecursively extracted {len(nested_extracted)} nested ZIP archive(s)."
                if nested_extracted else "\nNo nested ZIP archive was found to extract."
            )
            raise RuntimeError(str(exc) + nested_note + "\n" + diagnostics) from exc

        if len(images) != expected_count:
            raise RuntimeError(
                f"Internal dataset selection error: expected {expected_count}, selected {len(images)}."
            )
        source_collection = source_dir.relative_to(extracted_dir).as_posix()
        source_dir_text = source_dir.as_posix()

    prepared_png = processed_dir / "png"
    prepared_png.mkdir(parents=True, exist_ok=True)
    labels = deterministic_split(
        expected_count,
        seed=seed,
        train_count=6000 if expected_count == 10_000 else int(0.6 * expected_count),
        validation_count=2000 if expected_count == 10_000 else int(0.2 * expected_count),
        test_count=2000 if expected_count == 10_000 else expected_count - int(0.8 * expected_count),
    )

    rows: list[dict[str, object]] = []
    converted_count = 0
    reused_count = 0
    resolved_shape: tuple[int, int] | None = tuple(expected_shape) if expected_shape is not None else None

    zip_handle = zipfile.ZipFile(archive, "r") if archive_mode else None
    try:
        iterable = members if archive_mode else images
        for i, (item, split) in enumerate(zip(iterable, labels, strict=True), start=1):
            if archive_mode:
                member = str(item)
                assert zip_handle is not None
                arr = _read_u8_gray_bytes(zip_handle.read(member), member, strict_grayscale=True)
                source_id = Path(member).stem
                source_path_text = f"zip://{archive.as_posix()}!/{member}"
                source_format = Path(member).suffix.lower().lstrip(".")
            else:
                src = Path(item)
                arr = _read_u8_gray(src, strict_grayscale=True)
                source_id = src.stem
                source_path_text = src.as_posix()
                source_format = src.suffix.lower().lstrip(".")

            current_shape = tuple(int(v) for v in arr.shape)
            if resolved_shape is None:
                resolved_shape = current_shape
                print(f"Auto-detected source image shape: {resolved_shape[0]}x{resolved_shape[1]} pixels.")
            elif current_shape != resolved_shape:
                raise RuntimeError(
                    f"Inconsistent image shape {arr.shape} for {source_path_text}; "
                    f"expected all images to match {resolved_shape}."
                )

            dst = prepared_png / f"{source_id}.png"
            pixel_sha = _pixel_sha256(arr)

            can_reuse = False
            if _exists_with_retry(dst):
                try:
                    existing = _read_u8_gray(dst, strict_grayscale=True)
                    can_reuse = existing.shape == arr.shape and _pixel_sha256(existing) == pixel_sha
                except Exception:
                    can_reuse = False

            if can_reuse:
                reused_count += 1
            else:
                _write_verified_png(dst, arr)
                converted_count += 1

            rows.append(
                {
                    "source_id": source_id,
                    "split": split,
                    "source_path": source_path_text,
                    "source_format": source_format,
                    "source_collection": source_collection,
                    "source_is_lossy": source_is_lossy,
                    "source_access_mode": "zip-direct" if archive_mode else "extracted-files",
                    "prepared_path": dst.as_posix(),
                    "path": dst.as_posix(),
                    "width": int(arr.shape[1]),
                    "height": int(arr.shape[0]),
                    "dtype": str(arr.dtype),
                    "pixel_sha256": pixel_sha,
                }
            )

            if i % 500 == 0 or i == expected_count:
                print(f"Prepared {i:5d}/{expected_count} images...")
    finally:
        if zip_handle is not None:
            zip_handle.close()

    manifest_path = processed_dir / "dataset_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts = {name: sum(row["split"] == name for row in rows) for name in ("train", "validation", "test")}
    metadata = {
        "dataset": source_label,
        "canonical_bossbase_1_01": not source_is_lossy,
        "source_is_lossy": source_is_lossy,
        "source_collection": source_collection,
        "source_access_mode": "zip-direct" if archive_mode else "extracted-files",
        "archive_path": archive.as_posix(),
        "archive_sha256": archive_sha,
        "expected_count": expected_count,
        "actual_count": expected_count,
        "expected_shape": list(expected_shape) if expected_shape is not None else None,
        "actual_shape": list(resolved_shape) if resolved_shape is not None else None,
        "shape_policy": "explicit" if expected_shape is not None else "auto-detect-first-image-and-enforce-consistency",
        "pixel_type": "uint8 grayscale",
        "split_seed": seed,
        "split_counts": counts,
        "conversion": (
            "decoded source pixels -> 8-bit grayscale PNG; pixel equality verified after PNG reload. "
            "For JPEG input this preserves the decoded pixel array, not pre-JPEG sensor/image values."
        ),
        "manifest_path": manifest_path.as_posix(),
        "prepared_dir": prepared_png.as_posix(),
    }
    metadata_path = processed_dir / "dataset_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    legacy_manifest_path = processed_dir.parent / "dataset_manifest.csv"
    shutil.copy2(manifest_path, legacy_manifest_path)
    legacy_metadata_path = processed_dir.parent / "dataset_metadata.json"
    shutil.copy2(metadata_path, legacy_metadata_path)

    return BossbasePreparationResult(
        archive_path=archive.as_posix(),
        archive_sha256=archive_sha,
        extracted_dir=extracted_dir.as_posix(),
        source_dir=source_dir_text,
        source_label=source_label,
        source_is_lossy=source_is_lossy,
        prepared_dir=prepared_png.as_posix(),
        manifest_path=manifest_path.as_posix(),
        metadata_path=metadata_path.as_posix(),
        image_count=expected_count,
        train_count=counts["train"],
        validation_count=counts["validation"],
        test_count=counts["test"],
        converted_count=converted_count,
        reused_count=reused_count,
    )

def result_as_dict(result: BossbasePreparationResult) -> dict[str, object]:
    return asdict(result)
