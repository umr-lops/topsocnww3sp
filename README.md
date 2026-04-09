# topsocnww3sp

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

[![Coverage][coverage-badge]][coverage-link]

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

association rules between S1-OSW data / daily trackfiles / monthly WW3 field
output / monthly WW3 spectra files example: all these SAFE: ls
/home/datawork-cersat-public/cache/project/mpc-sentinel1/data/esa/sentinel-1a/L2/IW/S1A_IW_OCN**2S/2025/097/
S1A_IW_OCN**2SDV_20250407T032436_20250407T032501_058645_0742BF_C7D8.SAFE
S1A_IW_OCN**2SDV_20250407T114432_20250407T114501_058650_0742FD_EF9F.SAFE
S1A_IW_OCN**2SDV_20250407T213854_20250407T213923_058656_074336_D8A0.SAFE
S1A_IW_OCN**2SDV_20250407T032501_20250407T032526_058645_0742BF_165F.SAFE
S1A_IW_OCN**2SDV_20250407T114501_20250407T114526_058650_0742FD_8406.SAFE
S1A_IW_OCN**2SDV_20250407T213923_20250407T213948_058656_074336_EEDA.SAFE
S1A_IW_OCN**2SDV_20250407T032526_20250407T032551_058645_0742BF_B993.SAFE
S1A_IW_OCN\_\_2SDV_20250407T114526_20250407T114551_058650_0742FD_C9BA.SAFE

A SAFE contains netcdf files (.nc), for example:
s1a-iw1-osw-hh-20260405t102207-20260405t102237-063943-080b17-001.nc
s1a-iw3-osw-hh-20260405t102209-20260405t102239-063943-080b17-003.nc
s1a-iw2-osw-hh-20260405t102208-20260405t102238-063943-080b17-002.nc
s1a-iw-ocn-hh-20260405t102207-20260405t102237-063943-080B17-001.nc

a daily trackfile is named like this: trackfile_s1_iw_20260204.txt or
trackfile_s1_ew_20260204.txt it contains both intra and inter burst tiles. it
contains all the units together S1A+S1C+S1D for instance. The lines are ordered
in chronological order. Simultaneous acquisitions can occured with the differnt
Sentinel-1 unit.

Example of trackfile content: WAVEWATCH III TRACK LOCATIONS DATA 20250407 020615
-121.93000 35.00000 20250407 020615 -121.81000 35.01000 20250407 020615
-121.69000 35.03000 20250407 020615 -121.57000 35.05000 20250407 020615
-121.46000 35.06000 20250407 020615 -121.34000 35.08000 20250407 020615
-122.22000 35.13000 20250407 020615 -122.09000 35.14000 20250407 020615
-121.97000 35.16000 20250407 020615 -121.85000 35.18000 20250407 020615
-121.73000 35.20000 20250407 020615 -121.61000 35.21000 20250407 020615
-121.49000 35.23000 20250407 020615 -121.38000 35.25000 20250407 020615
-122.25000 35.29000 20250407 020615 -122.13000 35.31000 20250407 020615
-122.01000 35.33000 20250407 020615 -121.88000 35.35000 20250407 020615
-121.76000 35.36000 20250407 020615 -121.65000 35.38000 20250407 020615
-121.53000 35.40000 20250407 020615 -121.41000 35.41000 20250407 020616
-120.97000 33.83000 20250407 020616 -120.86000 33.84000 20250407 020616
-120.75000 33.86000 20250407 020616 -120.64000 33.87000 20250407 020616
-120.54000 33.89000 20250407 020616 -120.43000 33.90000 20250407 020616
-120.32000 33.92000 20250407 020616 -121.01000 33.99000 20250407 020616
-120.89000 34.01000 20250407 020616 -120.79000 34.02000 20250407 020616
-120.68000 34.04000 20250407 020616 -120.57000 34.05000 20250407 020616
-120.46000 34.07000 20250407 020616 -120.36000 34.08000 20250407 020616
-121.04000 34.16000 20250407 020616 -120.93000 34.17000 20250407 020616
-120.82000 34.19000

associated WW3 field output is store in:
/scale/project/wave/WW3/PROJECT/IRI/IRI_15KM_01/012504/IRIGLOB-7M/2025-04-07T00_2025-04-08T00/FIELD_NC/WW3-IRIGLOB-7M_202504.nc

associated WW3 spectra nc file is tore in:
/scale/project/wave/WW3/PROJECT/IRI/IRI_15KM_01/012504/IRIGLOB-7M/2025-04-07T00_2025-04-08T00/TRACK_NC/WW3-IRIGLOB-7M_202504_trck.nc
