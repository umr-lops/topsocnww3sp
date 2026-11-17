"""Unit tests for the l2c_sequential_processing batch runner."""

import logging
import sys
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


def test_has_track_data_flat_layout(tmp_path):
    """A TRACK_NC directory directly below the run directory is accepted."""
    run = tmp_path / "run"
    (run / "TRACK_NC").mkdir(parents=True)
    assert not seq._has_track_data(run)
    (run / "TRACK_NC" / "x_trck.nc").touch()
    assert seq._has_track_data(run)


def test_resolve_ww3_path(tmp_path):
    """Explicit file wins; no source gives None; a bad SAFE name a reason."""
    safe = tmp_path / SAFE_NAME
    assert seq.resolve_ww3_path(safe, "/a/ww3.nc", None) == (Path("/a/ww3.nc"), None)
    assert seq.resolve_ww3_path(safe, None, None) == (None, None)
    path, reason = seq.resolve_ww3_path(tmp_path / "bad.SAFE", None, str(tmp_path))
    assert path is None
    assert reason is not None
    assert "date" in reason


def test_init_worker_sets_synchronous_dask():
    """Workers run dask single-threaded (one process per core)."""
    import dask  # noqa: PLC0415

    seq._init_worker(logging.INFO)
    assert dask.config.get("scheduler") == "synchronous"
    dask.config.set(scheduler=None)


def test_batch_process_safes_process_pool(tmp_path):
    """With workers > 1 SAFEs run in a process pool; errors are collected."""
    config = tmp_path / "config.yml"
    config.write_text(yaml.safe_dump({"product_version": "v0.2"}))
    # SAFEs without measurement directory -> FileNotFoundError in the workers
    safes = [tmp_path / SAFE_NAME, tmp_path / SAFE_NAME.replace("711E", "AAAA")]
    for safe in safes:
        safe.mkdir()
    stats = seq.batch_process_safes(
        safe_dirs=safes,
        ww3_file=str(tmp_path / "ww3.nc"),
        ww3_dir=None,
        config=config,
        mode="lasso",
        output_dir=tmp_path / "out",
        verbose=True,
        overwrite=False,
        workers=2,
    )
    assert stats["failed_other"] == 2
    assert all("FileNotFoundError" in r for _, r in stats["failed_safes"])
    seq.print_summary(stats)


def _write_listing(tmp_path: Path, n: int) -> tuple[Path, list[Path]]:
    """Listing of n existing SAFEs written with an 'old platform' prefix."""
    root = tmp_path / "legacy"
    safes = []
    for i in range(n):
        safe = root / SAFE_NAME.replace("711E", f"{i:04d}")
        safe.mkdir(parents=True)
        safes.append(safe)
    listing = tmp_path / "listing.txt"
    listing.write_text("".join(f"/old/cache/{s.relative_to(root)}\n" for s in safes))
    return listing, safes


def test_main_chunk_and_path_replace(tmp_path, monkeypatch):
    """main() remaps the listing paths and keeps only chunk I of N."""
    listing, safes = _write_listing(tmp_path, 5)
    config = tmp_path / "config.yml"
    config.write_text(yaml.safe_dump({"product_version": "v0.2"}))
    seen: list[Path] = []

    def fake_process_safe(safe_dir, *_args, **_kwargs):
        seen.append(safe_dir)
        return [safe_dir / "x.nc"]

    monkeypatch.setattr(seq, "process_safe", fake_process_safe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "procl2c-sequential",
            "--listing-safe",
            str(listing),
            "--config",
            str(config),
            "--ww3-file",
            str(tmp_path / "ww3.nc"),
            "--output-dir",
            str(tmp_path / "out"),
            "--path-replace",
            f"/old/cache={tmp_path / 'legacy'}",
            "--chunk",
            "1/2",
            "--workers",
            "1",
        ],
    )
    seq.main()
    assert seen == safes[1::2]


@pytest.mark.parametrize(
    "extra",
    [
        ["--path-replace", "no-equal-sign"],
        ["--chunk", "one/two"],
        ["--ww3-dir", "/somewhere"],  # with --ww3-file: mutually exclusive
    ],
)
def test_main_invalid_arguments(tmp_path, monkeypatch, extra):
    """Invalid CLI arguments stop with exit code 1."""
    listing, _ = _write_listing(tmp_path, 1)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "procl2c-sequential",
            "--listing-safe",
            str(listing),
            "--config",
            "c.yml",
            "--ww3-file",
            "ww3.nc",
            "--output-dir",
            str(tmp_path / "out"),
            "--path-replace",
            f"/old/cache={tmp_path / 'legacy'}",
            *extra,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        seq.main()
    assert exc.value.code == 1


def test_main_dry_run(tmp_path, monkeypatch):
    """--dry-run lists the SAFEs and their WW3 run directory only."""
    listing, _ = _write_listing(tmp_path, 2)
    ww3_root = tmp_path / "ww3"
    _make_run_dir(ww3_root, "2026-03-31T00_2026-04-01T00")
    monkeypatch.setattr(
        seq, "process_safe", lambda *_, **__: pytest.fail("must not process")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "procl2c-sequential",
            "--listing-safe",
            str(listing),
            "--config",
            "c.yml",
            "--ww3-dir",
            str(ww3_root),
            "--output-dir",
            str(tmp_path / "out"),
            "--path-replace",
            f"/old/cache={tmp_path / 'legacy'}",
            "--dry-run",
        ],
    )
    seq.main()
