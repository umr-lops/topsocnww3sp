#!/usr/bin/env python3
"""Batch processor for multiple SAFE directories using l2c_processor."""

import argparse
import logging
import multiprocessing
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from topsocnww3sp.l2c_processor import process_safe

# WW3 run directory name, e.g. "2026-03-31T00_2026-04-01T00"
_WW3_RUN_DIR_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2})_(\d{4}-\d{2}-\d{2}T\d{2})$")

logger = logging.getLogger(__name__)


def parse_listing_file(
    listing_path: Path, path_replace: list[tuple[str, str]] | None = None
) -> list[Path]:
    """Parse a listing file containing SAFE directory paths.

    Args:
        listing_path: Path to the listing file (one SAFE path per line,
                      lines starting with # are ignored)
        path_replace: Optional (old, new) prefix substitutions applied to each
            path, e.g. to remap a listing made on another platform

    Returns:
        List of valid SAFE directory paths

    Raises:
        FileNotFoundError: If listing file does not exist
    """
    if not listing_path.exists():
        error_msg = f"Listing file not found: {listing_path}"
        raise FileNotFoundError(error_msg)

    safe_dirs: list[Path] = []
    with listing_path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            for old, new in path_replace or []:
                if line.startswith(old):
                    line = new + line[len(old) :]
            safe_path = Path(line)
            if safe_path.exists() and safe_path.is_dir():
                safe_dirs.append(safe_path)
            else:
                logger.warning("Skipping invalid SAFE directory: %s", line)

    return safe_dirs


def extract_sar_date(safe_dir: Path) -> datetime | None:
    """Extract acquisition date from SAFE directory name.

    SAFE name format: S1A_IW_OCN__2SDV_YYYYMMDDTHHMMSS_...SAFE
    Example: S1A_IW_OCN__2SDV_20220107T062432_20220107T062457_...SAFE

    Args:
        safe_dir: SAFE directory path

    Returns:
        Datetime object or None if parsing fails
    """
    name = safe_dir.name
    try:
        parts = name.split("_")
        for part in parts:
            if len(part) == 15 and "T" in part:  # YYYYMMDDTHHMMSS
                # Make datetime aware (UTC) to satisfy DTZ007
                return datetime.strptime(part, "%Y%m%dT%H%M%S").replace(
                    tzinfo=timezone.utc
                )
    except (ValueError, IndexError):
        pass
    return None


def find_matching_ww3_dirs(
    ww3_root: Path,
    sar_date: datetime,
) -> list[Path]:
    """Find WW3 run directories matching the SAR acquisition date.

    The WW3 directory structure is expected to be:
    {ww3_root}/{start}_{stop}/{subgrid}/{start}_{stop}/TRACK_NC/
    with {start}_{stop} like ``2026-03-31T00_2026-04-01T00``. The run
    directories whose [start, stop) range contains the SAR time are returned
    first; a name-based search on the date is used as a fallback.

    Args:
        ww3_root: Root directory containing WW3 run directories
        sar_date: SAR acquisition date

    Returns:
        List of paths to run directories (parent of subgrid), best match first
    """
    return list(_find_matching_ww3_dirs_cached(ww3_root, sar_date))


@lru_cache(maxsize=256)
def _find_matching_ww3_dirs_cached(
    ww3_root: Path, sar_date: datetime
) -> tuple[Path, ...]:
    """Cached implementation of find_matching_ww3_dirs (one scan per date)."""
    sar_utc = (
        sar_date.astimezone(timezone.utc)
        if sar_date.tzinfo
        else sar_date.replace(tzinfo=timezone.utc)
    )

    # 1. Run directories whose time range contains the SAR time
    in_range: list[Path] = []
    for run_dir in sorted(ww3_root.iterdir()):
        match = _WW3_RUN_DIR_RE.match(run_dir.name)
        if not match or not run_dir.is_dir():
            continue
        start, stop = (
            datetime.strptime(g, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
            for g in match.groups()
        )
        if start <= sar_utc < stop and _has_track_data(run_dir):
            in_range.append(run_dir)
    if in_range:
        return tuple(in_range)

    # 2. Fallback: directories named after the SAR date
    date_str = sar_date.strftime("%Y-%m-%d")
    candidates = {d for d in ww3_root.glob(f"*{date_str}T*") if d.is_dir()}
    valid = sorted(d for d in candidates if _has_track_data(d))
    for d in valid:
        logger.debug("Found valid WW3 directory: %s", d)
    return tuple(valid)


def _has_track_data(run_dir: Path) -> bool:
    """True if a TRACK_NC directory with .nc files exists below run_dir."""
    for track_dir in run_dir.glob("*/*/TRACK_NC"):
        if any(track_dir.glob("*.nc")):
            return True
    track_dir = run_dir / "TRACK_NC"
    return track_dir.is_dir() and any(track_dir.glob("*.nc"))


def dry_run_summary(
    safe_dirs: list[Path],
    ww3_dir: str | None,
    output_dir: Path,
    mode: str,
) -> None:
    """Print dry run summary without processing using logger."""
    logger.info("=" * 70)
    logger.info("DRY RUN SUMMARY")
    logger.info("=" * 70)
    logger.info("Mode: %s", mode)
    logger.info("Output directory: %s", output_dir)
    logger.info(
        "WW3 root directory: %s",
        ww3_dir or "Not specified (using explicit file)",
    )
    logger.info("-" * 70)
    logger.info("Total SAFE directories found: %d", len(safe_dirs))

    success_count = 0
    no_date_count = 0
    no_ww3_count = 0

    for safe_dir in safe_dirs:
        logger.info("📁 %s", safe_dir.name)

        sar_date = extract_sar_date(safe_dir)
        if sar_date is None:
            logger.info("   ❌ Could not extract date from SAFE name")
            no_date_count += 1
            continue

        logger.info(
            "   📅 Acquisition date: %s",
            sar_date.strftime("%Y-%m-%d %H:%M:%S"),
        )

        if ww3_dir:
            matching_dirs = find_matching_ww3_dirs(Path(ww3_dir), sar_date)
            if matching_dirs:
                logger.info("   🌊 WW3 directory: %s", matching_dirs[0])
                success_count += 1
            else:
                logger.info("   ❌ No matching WW3 directory found")
                no_ww3_count += 1
        else:
            logger.info("   🌊 WW3: Using explicit path (not checked in dry-run)")
            success_count += 1

        logger.info("")

    logger.info("-" * 70)
    logger.info("Summary:")
    logger.info("  ✅ Ready to process: %d", success_count)
    logger.info("  ⚠️  Could not extract date: %d", no_date_count)
    logger.info("  ❌ No WW3 directory found: %d", no_ww3_count)
    logger.info("=" * 70)
    logger.info(
        "\n⚠️  This was a DRY RUN. No processing was performed.\n"
        "   Remove --dry-run to execute actual processing.\n"
    )


def resolve_ww3_path(
    safe_dir: Path, ww3_file: str | None, ww3_dir: str | None
) -> tuple[Path | None, str | None]:
    """WW3 file/dir to use for a SAFE.

    Returns:
        (WW3 path, None) or (None, failure reason)
    """
    if ww3_file:
        return Path(ww3_file), None
    if not ww3_dir:
        return None, None
    sar_date = extract_sar_date(safe_dir)
    if sar_date is None:
        return None, "Could not extract date from SAFE name"
    matching_dirs = find_matching_ww3_dirs(Path(ww3_dir), sar_date)
    if not matching_dirs:
        return None, f"No WW3 directory for date {sar_date}"
    return matching_dirs[0], None


def _process_one_safe(
    safe_dir: Path,
    ww3_path: Path | None,
    config: dict[str, Any],
    mode: str,
    output_dir: Path,
    overwrite: bool,
) -> tuple[str, str, int]:
    """Process one SAFE, never raising.

    Returns:
        (SAFE path, status, number of files written); status is "ok",
        "no_colocation" or "error: <message>"
    """
    try:
        written = process_safe(
            safe_dir,
            config,
            output_dir,
            mode=mode,
            ww3_path=ww3_path,
            overwrite=overwrite,
        )
    except Exception as exc:  # a failing SAFE must not stop the batch
        logger.exception("Unexpected error processing %s", safe_dir.name)
        return str(safe_dir), f"error: {type(exc).__name__}: {exc}"[:300], 0
    return str(safe_dir), "ok" if written else "no_colocation", len(written)


def _init_worker(log_level: int) -> None:
    """Configure logging and single-threaded numerics in a worker process."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
    )
    try:
        import dask  # noqa: PLC0415 - optional, only to tune the scheduler

        dask.config.set(scheduler="synchronous")
    except ImportError:
        pass


def batch_process_safes(
    safe_dirs: list[Path],
    ww3_file: str | None,
    ww3_dir: str | None,
    config: Path,
    mode: str,
    output_dir: Path,
    verbose: bool,
    overwrite: bool,
    workers: int = 1,
) -> dict[str, Any]:
    """Process multiple SAFE directories and collect statistics.

    Args:
        safe_dirs: SAFE directories to process
        ww3_file: WW3 file or run directory used for every SAFE
        ww3_dir: WW3 root directory, the run directory is found per SAFE date
        config: Path to the YAML config
        mode: Matching mode
        output_dir: Base output directory
        verbose: Debug logging
        overwrite: Overwrite existing outputs
        workers: Number of SAFEs processed in parallel (processes)

    Returns:
        Statistics dictionary
    """
    with config.open(encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    stats: dict[str, Any] = {
        "total_safes": len(safe_dirs),
        "processed_successfully": 0,
        "no_colocation": 0,
        "failed_no_ww3_dir": 0,
        "failed_other": 0,
        "files_written": 0,
        "successful_safes": [],
        "failed_safes": [],
    }

    jobs: list[tuple[Path, Path | None]] = []
    for safe_dir in safe_dirs:
        ww3_path, reason = resolve_ww3_path(safe_dir, ww3_file, ww3_dir)
        if reason is not None:
            logger.warning("%s: %s", safe_dir.name, reason)
            stats["failed_no_ww3_dir"] += 1
            stats["failed_safes"].append((str(safe_dir), reason))
            continue
        jobs.append((safe_dir, ww3_path))

    def record(result: tuple[str, str, int], index: int) -> None:
        safe, status, n_written = result
        logger.info(
            "[%d/%d] %s: %s (%d files)",
            index,
            len(jobs),
            Path(safe).name,
            status,
            n_written,
        )
        stats["files_written"] += n_written
        if status == "ok":
            stats["processed_successfully"] += 1
            stats["successful_safes"].append(safe)
        elif status == "no_colocation":
            stats["no_colocation"] += 1
        else:
            stats["failed_other"] += 1
            stats["failed_safes"].append((safe, status))

    args = (config_dict, mode, output_dir, overwrite)
    if workers <= 1:
        for i, (safe_dir, ww3_path) in enumerate(jobs, 1):
            logger.info("=" * 70)
            logger.info("Processing SAFE %d/%d: %s", i, len(jobs), safe_dir.name)
            record(_process_one_safe(safe_dir, ww3_path, *args), i)
        return stats

    log_level = logging.DEBUG if verbose else logging.INFO
    ctx = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(log_level,),
    ) as pool:
        futures = [
            pool.submit(_process_one_safe, safe_dir, ww3_path, *args)
            for safe_dir, ww3_path in jobs
        ]
        for i, fut in enumerate(as_completed(futures), 1):
            record(fut.result(), i)
    return stats


def print_summary(stats: dict[str, Any]) -> None:
    """Print processing summary statistics using logger."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info("=" * 70)
    logger.info("Total SAFE directories processed: %d", stats["total_safes"])
    logger.info("Successfully processed: %d", stats["processed_successfully"])
    logger.info("L2C files written: %d", stats["files_written"])
    logger.info(
        "No co-location (land only, no WW3 match, or outputs already present): %d",
        stats["no_colocation"],
    )
    logger.info("Failed - no WW3 directory found: %d", stats["failed_no_ww3_dir"])
    logger.info("Failed - other errors: %d", stats["failed_other"])
    logger.info("-" * 70)

    if stats["successful_safes"]:
        logger.info("\n✅ Successful SAFEs:")
        for safe_path in stats["successful_safes"]:
            logger.info("  - %s", Path(safe_path).name)

    if stats["failed_safes"]:
        logger.info("\n❌ Failed SAFEs:")
        for safe_path, reason in stats["failed_safes"]:
            logger.info("  - %s: %s", Path(safe_path).name, reason)

    logger.info("=" * 70)


def main() -> None:
    """Main entry point for batch processor."""
    parser = argparse.ArgumentParser(
        description="Batch process multiple SAFE directories using l2c_processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using explicit WW3 file/directory
  %(prog)s --listing-safe safes.txt --config config.yml --mode lasso --output-dir ./output --ww3-file /path/to/ww3

  # Using automatic WW3 discovery from root directory
  %(prog)s --listing-safe safes.txt --config config.yml --mode lasso --output-dir ./output --ww3-dir /scale/project/wave/WW3/PROJECT/IRI/IRI_15KM_01/

  # Dry run to check what would be processed
  %(prog)s --listing-safe safes.txt --config config.yml --mode lasso --output-dir ./output --ww3-dir /path/to/ww3 --dry-run

  # With verbose output and overwrite
  %(prog)s --listing-safe safes.txt --config config.yml --mode lasso --ww3-dir /path/to/ww3/root --overwrite --verbose
        """,
    )
    parser.add_argument(
        "--listing-safe",
        required=True,
        help="Path to text file containing SAFE directory paths (one per line)",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file with thresholds and directories",
    )
    parser.add_argument(
        "--mode",
        choices=["1to1", "unique", "many", "lasso"],
        default="lasso",
        help="Matching mode (default: lasso)",
    )
    parser.add_argument(
        "--ww3-file",
        default=None,
        help="Direct path to WW3 file or directory (legacy mode)",
    )
    parser.add_argument(
        "--ww3-dir",
        default=None,
        help="Root directory for automatic WW3 discovery (replaces --ww3-file)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Base directory to save output files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        help="Number of SAFEs processed in parallel "
        "(default: $SLURM_CPUS_PER_TASK, else 1)",
    )
    parser.add_argument(
        "--chunk",
        default=None,
        help="Process only chunk I of N of the listing, as 'I/N' (0-based I), "
        "e.g. '$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT' in a Slurm job array",
    )
    parser.add_argument(
        "--path-replace",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Replace the OLD prefix of the SAFE paths of the listing by NEW "
        "(repeatable), e.g. /home/datawork-cersat-public/cache=/legacy/project/cersat/public/cache",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list what would be processed without actually running the processor",
    )
    args = parser.parse_args()

    # Validate WW3 arguments
    if args.ww3_file and args.ww3_dir:
        logger.error("Cannot specify both --ww3-file and --ww3-dir")
        sys.exit(1)
    if not args.ww3_file and not args.ww3_dir:
        logger.error("Either --ww3-file or --ww3-dir must be specified")
        sys.exit(1)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("topsocnww3sp.l2c_processor").setLevel(log_level)

    listing_path = Path(args.listing_safe)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)

    path_replace: list[tuple[str, str]] = []
    for item in args.path_replace:
        old, sep, new = item.partition("=")
        if not sep or not old:
            logger.error("Invalid --path-replace %r, expected OLD=NEW", item)
            sys.exit(1)
        path_replace.append((old, new))

    logger.info("Reading listing file: %s", listing_path)
    safe_dirs = parse_listing_file(listing_path, path_replace)
    if args.chunk:
        try:
            i_chunk, n_chunk = (int(v) for v in args.chunk.split("/"))
        except ValueError:
            logger.exception("Invalid --chunk %r, expected I/N", args.chunk)
            sys.exit(1)
        safe_dirs = safe_dirs[i_chunk::n_chunk]
        logger.info("Chunk %d/%d: %d SAFEs", i_chunk, n_chunk, len(safe_dirs))

    if not safe_dirs:
        logger.error("No valid SAFE directories found in listing file")
        sys.exit(1)

    logger.info("Found %d SAFE directories to process", len(safe_dirs))

    if args.dry_run:
        dry_run_summary(safe_dirs, args.ww3_dir, output_dir, args.mode)
        return

    start_time = time.time()
    stats = batch_process_safes(
        safe_dirs=safe_dirs,
        ww3_file=args.ww3_file,
        ww3_dir=args.ww3_dir,
        config=config_path,
        mode=args.mode,
        output_dir=output_dir,
        verbose=args.verbose,
        overwrite=args.overwrite,
        workers=args.workers,
    )
    elapsed_time = time.time() - start_time

    print_summary(stats)
    logger.info("Total processing time: %.2f seconds", elapsed_time)


if __name__ == "__main__":
    main()
