"""Unit tests for the l2c_sequential_processing batch runner."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from topsocnww3sp import l2c_sequential_processing as seq

SAFE_NAME = "S1A_IW_OCN__2SDV_20260331T101354_20260331T101419_063870_080855_711E.SAFE"


def _make_run_dir(root: Path, name: str) -> Path:
    """WW3 run directory with one grid TRACK_NC file."""
    run = root / name
    track = run / "IRIGLOB-7M" / name / "TRACK_NC"
    track.mkdir(parents=True)
    (track / "WW3-IRIGLOB-7M_202603_trck.nc").touch()
    return run


@pytest.fixture(autouse=True)
def _clear_cache():
    seq._find_matching_ww3_dirs_cached.cache_clear()


def test_find_matching_ww3_dirs_picks_run_containing_sar_time(tmp_path):
    """The run whose [start, stop) range holds the SAR time is chosen."""
    _make_run_dir(tmp_path, "2026-03-30T00_2026-03-31T00")
    good = _make_run_dir(tmp_path, "2026-03-31T00_2026-04-01T00")
    _make_run_dir(tmp_path, "2026-04-01T00_2026-04-02T00")
    (tmp_path / "2026-03-31T00_empty").mkdir()
    sar = datetime(2026, 3, 31, 10, 13, 54, tzinfo=timezone.utc)
    assert seq.find_matching_ww3_dirs(tmp_path, sar) == [good]


def test_find_matching_ww3_dirs_none(tmp_path):
    """No run directory for the date gives an empty list."""
    _make_run_dir(tmp_path, "2025-04-07T00_2025-04-08T00")
    sar = datetime(2026, 3, 31, 10, 13, 54, tzinfo=timezone.utc)
    assert seq.find_matching_ww3_dirs(tmp_path, sar) == []


def test_parse_listing_file_path_replace(tmp_path):
    """Listing paths are remapped with --path-replace before being checked."""
    new_root = tmp_path / "legacy"
    safe = new_root / "esa" / SAFE_NAME
    safe.mkdir(parents=True)
    listing = tmp_path / "listing.txt"
    listing.write_text(
        f"# comment\n/home/datawork/cache/esa/{SAFE_NAME}\n/home/datawork/cache/esa/missing.SAFE\n"
    )
    assert seq.parse_listing_file(listing) == []
    got = seq.parse_listing_file(listing, [("/home/datawork/cache", str(new_root))])
    assert got == [safe]


def test_extract_sar_date():
    """The SAFE start time is parsed as UTC."""
    assert seq.extract_sar_date(Path(SAFE_NAME)) == datetime(
        2026, 3, 31, 10, 13, 54, tzinfo=timezone.utc
    )


def test_batch_process_safes_statistics(tmp_path, monkeypatch):
    """Statuses of process_safe are counted; errors do not stop the batch."""
    config = tmp_path / "config.yml"
    config.write_text(yaml.safe_dump({"product_version": "v0.2"}))
    ww3_root = tmp_path / "ww3"
    _make_run_dir(ww3_root, "2026-03-31T00_2026-04-01T00")
    safes = [
        tmp_path / SAFE_NAME,
        tmp_path / SAFE_NAME.replace("711E", "AAAA"),
        tmp_path / SAFE_NAME.replace("711E", "BBBB"),
        tmp_path / SAFE_NAME.replace("20260331T101354", "20250101T000000"),
    ]

    def fake_process_safe(safe_dir, _config, output_dir, **_kwargs):
        if "AAAA" in safe_dir.name:
            return []
        if "BBBB" in safe_dir.name:
            msg = "boom"
            raise RuntimeError(msg)
        return [output_dir / "a.nc", output_dir / "b.nc"]

    monkeypatch.setattr(seq, "process_safe", fake_process_safe)
    stats = seq.batch_process_safes(
        safe_dirs=safes,
        ww3_file=None,
        ww3_dir=str(ww3_root),
        config=config,
        mode="lasso",
        output_dir=tmp_path / "out",
        verbose=False,
        overwrite=False,
        workers=1,
    )
    assert stats["processed_successfully"] == 1
    assert stats["files_written"] == 2
    assert stats["no_colocation"] == 1
    assert stats["failed_other"] == 1
    assert stats["failed_no_ww3_dir"] == 1
    assert any("boom" in reason for _, reason in stats["failed_safes"])
