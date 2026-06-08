# topsocnww3sp

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]
[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]
[![Coverage][coverage-badge]][coverage-link]

Python library to co-localize Sentinel-1 OCN Level-2 OSW (Ocean Swell Wave)
products with WW3 (WaveWatch III) spectral data.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Modes of Operation](#modes-of-operation)
  - [Output Structure](#output-structure)
- [Visualization Tools](#visualization-tools)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Overview

The `topsocnww3sp` package provides tools to associate Sentinel-1 OSW (Ocean
Swell Wave) data with WW3 (WaveWatch III) wave model spectra. The
co-localization can be performed using different matching strategies depending
on the scientific use case.

![Example of 1to1 SAR/WW3 colocation](docs/coloc_1to1_llustration.png)

## Features

- **Multiple matching modes**:
  - `1to1`: One-to-one matching (closest WW3 point per SAR tile)
  - `unique`: Multiple SAR tiles can share the same WW3 point (pointer array)
  - `many`: Many-to-many mapping table (all matches within radius)
  - `lasso`: All WW3 points within buffered SAR subswath footprint

- **Multi-grid WW3 support**: Handles IRI configuration with Arctic, Antarctic
  and mid-latitude grids

- **CF-compliant NetCDF output**: L2C products with full metadata and provenance
  tracking

- **Interactive visualization**: GIF animations and static maps for validation

- **SAR SAFE processing**: Process entire SAFE directories (all subswaths
  IW1/IW2/IW3 or EW1..EW5) in one run

## Installation

### From PyPI

```bash
pip install topsocnww3sp
```

### From Conda-Forge

```bash
conda install -c conda-forge topsocnww3sp
```

### For development

```bash
git clone https://github.com/umr-lops/topsocnww3sp.git
cd topsocnww3sp
pip install -e ".[dev]"
```

## Configuration

Create a `config.yml` file with the following structure:

```yaml
# WW3 data directory
directory_ww3spectra_output: /path/to/ww3/data

# Temporal and spatial thresholds
TIME_THRESHOLD_MINUTES: 30
DISTANCE_THRESHOLD_KM: 20
BUFFER_DEG: 0.1

# Product version (will be appended to output filename)
product_version: "v1.0"

# Multi-grid WW3 configuration (for IRI grids)
ww3_grids:
  arctic:
    pattern: "ARC-*/YYYY-*/TRACK_NC/WW3-ARC-*_*_trck.nc"
  antarctic:
    pattern: "ANTARC-*/YYYY-*/TRACK_NC/WW3-ANTARC-*_*_trck.nc"
  midlatitude:
    pattern: "IRIGLOB-*/YYYY-*/TRACK_NC/WW3-IRIGLOB-*_*_trck.nc"
```

## Usage

### Command Line Interface

#### SAFE directory processing (all subswaths)

```bash
procl2c \
  --ocn-safe /path/to/S1A_IW_OCN__2SDV_*.SAFE \
  --config config.yml \
  --mode lasso \
  --output-dir ./output \
  --overwrite
```

### Modes of Operation

| Mode     | Description                                             | Output Structure                                |
| -------- | ------------------------------------------------------- | ----------------------------------------------- |
| `1to1`   | Closest WW3 point per SAR tile                          | `WW3_{group}` with dimension `all_tiles`        |
| `unique` | Multiple SAR tiles can share WW3 point                  | Pointer array in `MATCH_MAP`                    |
| `many`   | All WW3 points within distance threshold                | Pair table with `sar_index` and `ww3_index`     |
| `lasso`  | All WW3 points within buffered footprint. Default mode. | Single `WW3` group per subswath, no `MATCH_MAP` |

### Output Structure

For a SAFE directory, the output files are organized as:

```
output/
└── YYYY/
    └── MM/
        └── DD/
            └── SAFE_NAME/
                ├── s1a-iw1-osw-..._v1.0.nc
                ├── s1a-iw2-osw-..._v1.0.nc
                └── s1a-iw3-osw-..._v1.0.nc
```

Each output NetCDF file contains:

- `SAR_intraburst` group: Original SAR OSW data
- `SAR_interburst` group: Interburst SAR data
- `WW3` group: Filtered WW3 spectra (lasso mode)
- `MATCH_MAP` group: Association table (other modes)

## Visualization Tools

### Interactive GIF animation

```bash
python scripts/l2c_map_gif.py \
  --l2c-file /path/to/L2C_lasso.nc \
  --tile 10 \
  --output colocation.gif
```

### Static maps for SAFE validation

```bash
python scripts/l2c_map_safe.py \
  --safe-dir /path/to/SAFE \
  --output-dir ./maps \
  --tile-zoom 8
```

Generated maps:

- Map 1: SAR intraburst tiles only
- Map 2: SAR intraburst + interburst tiles
- Map 3: SAR tiles + WW3 spectra positions

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=topsocnww3sp

# Run specific test
pytest tests/test_l2c_processor.py
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Run pre-commit hooks: `pre-commit run --all-files`
4. Submit a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{topsocnww3sp,
  author = {Grouazel, Antoine},
  title = {topsocnww3sp: Sentinel-1 OSW and WW3 co-localization},
  year = {2024},
  url = {https://github.com/umr-lops/topsocnww3sp}
}
```

## Acknowledgments

- Ifremer / LOPS laboratory
- ESA for Sentinel-1 data
- WW3 model developers

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/umr-lops/topsocnww3sp/actions/workflows/ci.yml/badge.svg
[actions-link]:             https://github.com/umr-lops/topsocnww3sp/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/topsocnww3sp
[conda-link]:               https://github.com/conda-forge/topsocnww3sp-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/umr-lops/topsocnww3sp/discussions
[pypi-link]:                https://pypi.org/project/topsocnww3sp/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/topsocnww3sp
[pypi-version]:             https://img.shields.io/pypi/v/topsocnww3sp
[rtd-badge]:                https://readthedocs.org/projects/topsocnww3sp/badge/?version=latest
[rtd-link]:                 https://topsocnww3sp.readthedocs.io/en/latest/?badge=latest
[coverage-badge]:           https://codecov.io/github/umr-lops/topsocnww3sp/branch/main/graph/badge.svg
[coverage-link]:            https://codecov.io/github/umr-lops/topsocnww3sp

<!-- prettier-ignore-end -->
