from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from rdhlab.dataset import deterministic_split, find_bossbase_archive, prepare_bossbase


def _make_toy_archive(tmp_path: Path, n: int = 10) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    for i in range(1, n + 1):
        arr = np.full((8, 8), i, dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(src / f"{i}.pgm")
    archive = tmp_path / "BOSSbase_1.01.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src.glob("*.pgm")):
            zf.write(p, arcname=f"BOSSbase_1.01/{p.name}")
    return archive


def test_deterministic_split_reproducible() -> None:
    a = deterministic_split(10, seed=7, train_count=6, validation_count=2, test_count=2)
    b = deterministic_split(10, seed=7, train_count=6, validation_count=2, test_count=2)
    assert a == b
    assert a.count("train") == 6
    assert a.count("validation") == 2
    assert a.count("test") == 2


def test_find_bossbase_archive_prefers_name(tmp_path: Path) -> None:
    (tmp_path / "other.zip").write_bytes(b"x")
    preferred = tmp_path / "BOSSbase_1.01.zip"
    preferred.write_bytes(b"y")
    assert find_bossbase_archive(tmp_path) == preferred


def test_prepare_toy_bossbase_lossless(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    archive = _make_toy_archive(raw, n=10)
    processed = tmp_path / "processed"

    result = prepare_bossbase(
        raw_dir=raw,
        processed_dir=processed,
        archive_path=archive,
        expected_count=10,
        expected_shape=(8, 8),
        seed=11,
    )

    assert result.image_count == 10
    assert result.train_count == 6
    assert result.validation_count == 2
    assert result.test_count == 2

    rows = list(csv.DictReader(Path(result.manifest_path).open(encoding="utf-8")))
    assert len(rows) == 10
    for row in rows:
        dst = np.asarray(Image.open(row["prepared_path"]), dtype=np.uint8)
        expected = np.full((8, 8), int(row["source_id"]), dtype=np.uint8)
        assert np.array_equal(expected, dst)


def test_prepare_nested_bossbase_zip(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()

    inner_src = tmp_path / "inner_src"
    inner_src.mkdir()
    for i in range(1, 11):
        arr = np.full((8, 8), i, dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(inner_src / f"{i}.pgm")

    inner_zip = tmp_path / "BOSSbase_1.01_inner.zip"
    with zipfile.ZipFile(inner_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for image in sorted(inner_src.glob("*.pgm")):
            zf.write(image, arcname=f"BOSSbase_1.01/{image.name}")

    outer_zip = raw / "archive.zip"
    with zipfile.ZipFile(outer_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(inner_zip, arcname="dataset/BOSSbase_1.01.zip")

    result = prepare_bossbase(
        raw_dir=raw,
        processed_dir=tmp_path / "processed",
        archive_path=outer_zip,
        expected_count=10,
        expected_shape=(8, 8),
        seed=11,
    )

    assert result.image_count == 10
    assert result.train_count == 6
    assert result.validation_count == 2
    assert result.test_count == 2


def _make_jpeg_variant_bundle(tmp_path: Path, n: int = 10) -> Path:
    root = tmp_path / "jpeg_bundle"
    for qf in (75, 90, 95):
        d = root / "grayscale" / f"QF{qf}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(1, n + 1):
            # Mildly structured arrays so JPEG remains valid grayscale input.
            arr = np.full((8, 8), (i * 11 + qf) % 255, dtype=np.uint8)
            Image.fromarray(arr, mode="L").save(d / f"{i}.jpg", quality=qf)
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(root.rglob("*.jpg")):
            zf.write(p, arcname=p.relative_to(root).as_posix())
    return archive


def test_prepare_jpeg_bundle_auto_highest_qf(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    archive = _make_jpeg_variant_bundle(raw, n=10)
    processed = tmp_path / "processed"

    result = prepare_bossbase(
        raw_dir=raw,
        processed_dir=processed,
        archive_path=archive,
        expected_count=10,
        expected_shape=(8, 8),
        seed=11,
        source_subdir="AUTO_HIGHEST_QF",
        allow_lossy_source=True,
    )

    assert result.image_count == 10
    assert result.source_is_lossy is True
    assert "QF95" in result.source_dir
    rows = list(csv.DictReader(Path(result.manifest_path).open(encoding="utf-8")))
    assert len(rows) == 10
    assert {row["source_collection"] for row in rows} == {"grayscale/QF95"}
    assert {row["source_is_lossy"] for row in rows} == {"True"}


def test_jpeg_bundle_requires_explicit_lossy_opt_in(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    archive = _make_jpeg_variant_bundle(raw, n=10)
    try:
        prepare_bossbase(
            raw_dir=raw,
            processed_dir=tmp_path / "processed",
            archive_path=archive,
            expected_count=10,
            expected_shape=(8, 8),
            seed=11,
            source_subdir="AUTO_HIGHEST_QF",
            allow_lossy_source=False,
        )
    except RuntimeError as exc:
        assert "lossy" in str(exc).lower() or "jpeg" in str(exc).lower()
    else:
        raise AssertionError("Lossy source should require explicit opt-in")


def test_jpeg_bundle_uses_direct_zip_streaming(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    archive = _make_jpeg_variant_bundle(raw, n=10)
    processed = tmp_path / "processed"

    result = prepare_bossbase(
        raw_dir=raw,
        processed_dir=processed,
        archive_path=archive,
        expected_count=10,
        expected_shape=(8, 8),
        seed=11,
        source_subdir="AUTO_HIGHEST_QF",
        allow_lossy_source=True,
    )

    rows = list(csv.DictReader(Path(result.manifest_path).open(encoding="utf-8")))
    assert {row["source_access_mode"] for row in rows} == {"zip-direct"}
    assert all(row["source_path"].startswith("zip://") for row in rows)
    assert "QF95" in result.source_dir
    # Direct ZIP mode must not need the 30k-file extraction directory.
    assert not (raw / "BOSSbase_1.01_extracted").exists()


def test_prepare_auto_detects_native_shape(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    archive = _make_jpeg_variant_bundle(raw, n=10)
    processed = tmp_path / "processed"

    result = prepare_bossbase(
        raw_dir=raw,
        processed_dir=processed,
        archive_path=archive,
        expected_count=10,
        expected_shape=None,
        seed=11,
        source_subdir="AUTO_HIGHEST_QF",
        allow_lossy_source=True,
    )
    rows = list(csv.DictReader(Path(result.manifest_path).open(encoding="utf-8")))
    assert {row["width"] for row in rows} == {"8"}
    assert {row["height"] for row in rows} == {"8"}

