#!/usr/bin/env python3
"""Batch processor for multiple SAFE directories using l2c_processor."""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from topsocnww3sp.l2c_processor import main as l2c_main

logger = logging.getLogger(__name__)


def parse_listing_file(listing_path: Path) -> list[Path]:
    """Parse a listing file containing SAFE directory paths.

    Args:
        listing_path: Path to the listing file (one SAFE path per line,
                      lines starting with # are ignored)

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
    """Find WW3 subdirectories matching the SAR acquisition date.

    The WW3 directory structure is expected to be:
    {ww3_root}/{date_range}/{subgrid}/{date_range}/TRACK_NC/
    or {ww3_root}/{date_range}/TRACK_NC/

    Args:
        ww3_root: Root directory containing WW3 subgrid directories
        sar_date: SAR acquisition date

    Returns:
        List of paths to date_range directories (parent of subgrid)
    """
    date_str = sar_date.strftime("%Y-%m-%d")
    year = sar_date.strftime("%Y")

    patterns = [
        f"*{date_str}T*",
        f"*{year}-*",
        f"*{year}*",
    ]

    date_dirs: list[Path] = []
    for pattern in patterns:
        # PERF401: use extend with generator
        date_dirs.extend(d for d in ww3_root.glob(pattern) if d.is_dir())

    # Also search recursively for date directories
    for path in ww3_root.rglob(f"*{date_str}T*"):
        if path.is_dir() and path not in date_dirs:
            date_dirs.append(path)

    # Filter and validate
    valid_dirs: list[Path] = []
    for date_dir in set(date_dirs):
        has_data = False

        # Look for TRACK_NC directories
        for track_dir in date_dir.rglob("TRACK_NC"):
            if track_dir.is_dir() and list(track_dir.glob("*.nc")):
                has_data = True
                break

        if not has_data:
            for subdir in date_dir.iterdir():
                if subdir.is_dir():
                    track_dir = subdir / "TRACK_NC"
                    if track_dir.exists() and list(track_dir.glob("*.nc")):
                        has_data = True
                        break

        if has_data:
            valid_dirs.append(date_dir)
            logger.debug("Found valid WW3 directory: %s", date_dir)
        else:
            logger.debug("No WW3 data in: %s", date_dir)

    return sorted(set(valid_dirs))


def create_processor_args(
    safe_dir: Path,
    ww3_path: str | Path | None,
    config: Path,
    mode: str,
    output_dir: Path,
    verbose: bool,
    overwrite: bool,
) -> argparse.Namespace:
    """Create arguments Namespace for l2c_processor.main().

    Args:
        safe_dir: SAFE directory to process
        ww3_path: WW3 file or directory path
        config: Path to config YAML
        mode: Processing mode (1to1, unique, many, lasso)
        output_dir: Base output directory
        verbose: Enable verbose logging
        overwrite: Overwrite existing files

    Returns:
        Namespace object with arguments
    """
    return argparse.Namespace(
        ocn_safe=str(safe_dir),
        ww3_file=str(ww3_path) if ww3_path else None,
        config=str(config),
        mode=mode,
        verbose=verbose,
        output_dir=str(output_dir),
        overwrite=overwrite,
    )


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


def batch_process_safes(
    safe_dirs: list[Path],
    ww3_file: str | None,
    ww3_dir: str | None,
    config: Path,
    mode: str,
    output_dir: Path,
    verbose: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Process multiple SAFE directories and collect statistics."""
    stats: dict[str, Any] = {
        "total_safes": len(safe_dirs),
        "processed_successfully": 0,
        "failed_temporal_match": 0,
        "failed_spatial_match": 0,
        "failed_no_ww3_dir": 0,
        "failed_no_ocean_tiles": 0,
        "failed_other": 0,
        "successful_safes": [],
        "failed_safes": [],
    }

    for i, safe_dir in enumerate(safe_dirs, 1):
        logger.info("=" * 70)
        logger.info("Processing SAFE %d/%d: %s", i, len(safe_dirs), safe_dir.name)
        logger.info("=" * 70)

        current_ww3_path: str | Path | None = None
        if ww3_file:
            current_ww3_path = ww3_file
        elif ww3_dir:
            sar_date = extract_sar_date(safe_dir)
            if sar_date is None:
                logger.warning(
                    "Could not extract date from SAFE name: %s", safe_dir.name
                )
                stats["failed_no_ww3_dir"] += 1
                stats["failed_safes"].append(
                    (str(safe_dir), "Could not extract date from SAFE name")
                )
                continue

            matching_dirs = find_matching_ww3_dirs(Path(ww3_dir), sar_date)
            if not matching_dirs:
                logger.warning("No matching WW3 directory found for %s", safe_dir.name)
                stats["failed_no_ww3_dir"] += 1
                stats["failed_safes"].append(
                    (str(safe_dir), f"No WW3 directory for date {sar_date}")
                )
                continue

            current_ww3_path = matching_dirs[0]
            logger.info("Using WW3 directory: %s", current_ww3_path)

        original_argv = sys.argv.copy()

        try:
            args = create_processor_args(
                safe_dir=safe_dir,
                ww3_path=current_ww3_path,
                config=config,
                mode=mode,
                output_dir=output_dir,
                verbose=verbose,
                overwrite=overwrite,
            )

            # Build command line arguments
            sys.argv = ["l2c_processor.py"]
            for key, value in vars(args).items():
                if value is None:
                    continue
                if key == "ww3_file" and value is not None:
                    sys.argv.append("--ww3-file")
                    sys.argv.append(str(value))
                elif key == "overwrite" and value:
                    sys.argv.append("--overwrite")
                elif key == "verbose" and value:
                    sys.argv.append("--verbose")
                elif key == "ocn_safe":
                    sys.argv.append("--ocn-safe")
                    sys.argv.append(str(value))
                elif key == "config":
                    sys.argv.append("--config")
                    sys.argv.append(str(value))
                elif key == "mode":
                    sys.argv.append("--mode")
                    sys.argv.append(value)
                elif key == "output_dir":
                    sys.argv.append("--output-dir")
                    sys.argv.append(str(value))

            l2c_main()

            stats["processed_successfully"] += 1
            stats["successful_safes"].append(str(safe_dir))

        except SystemExit as e:
            if e.code == 0:
                stats["processed_successfully"] += 1
                stats["successful_safes"].append(str(safe_dir))
            else:
                stats["failed_other"] += 1
                stats["failed_safes"].append((str(safe_dir), f"exit code {e.code}"))
                logger.exception(
                    "Failed to process %s (exit code %s)", safe_dir.name, e.code
                )
        except (OSError, ValueError, TypeError) as e:
            error_msg = str(e)
            if "No WW3 data within time window" in error_msg:
                stats["failed_temporal_match"] += 1
                logger.warning("No temporal match for %s", safe_dir.name)
            elif "No WW3 points inside the buffered subswath footprint" in error_msg:
                stats["failed_spatial_match"] += 1
                logger.warning("No spatial match for %s", safe_dir.name)
            elif "must supply at least one object to concatenate" in error_msg:
                stats["failed_no_ocean_tiles"] += 1
                logger.warning(
                    "No ocean tiles found (all tiles on land) for %s",
                    safe_dir.name,
                )
                stats["failed_safes"].append(
                    (str(safe_dir), "No ocean tiles (all tiles on land)")
                )
            else:
                stats["failed_other"] += 1
                logger.exception("Unexpected error processing %s", safe_dir.name)
                stats["failed_safes"].append((str(safe_dir), error_msg[:200]))

        finally:
            sys.argv = original_argv

    return stats


def print_summary(stats: dict[str, Any]) -> None:
    """Print processing summary statistics using logger."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info("=" * 70)
    logger.info("Total SAFE directories processed: %d", stats["total_safes"])
    logger.info("Successfully processed: %d", stats["processed_successfully"])
    logger.info("Failed - temporal match: %d", stats["failed_temporal_match"])
    logger.info("Failed - spatial match: %d", stats["failed_spatial_match"])
    logger.info("Failed - no WW3 directory found: %d", stats["failed_no_ww3_dir"])
    logger.info(
        "Failed - no ocean tiles (all on land): %d",
        stats["failed_no_ocean_tiles"],
    )
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

    logger.info("Reading listing file: %s", listing_path)
    safe_dirs = parse_listing_file(listing_path)

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
    )
    elapsed_time = time.time() - start_time

    print_summary(stats)
    logger.info("Total processing time: %.2f seconds", elapsed_time)


if __name__ == "__main__":
    main()
