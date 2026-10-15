Exploration of Different Strategies for the L2C Processor
==========================================================

This document explores different strategies for the L2C processor to determine
which WW3 spectra to associate with each track point, and how to handle cases
where no spectra are found within the defined spatiotemporal thresholds. This
includes testing different distance and time thresholds, as well as fallback
strategies such as using the nearest spectrum or interpolating between nearby
spectra.

During development, we have explored four different strategies for associating
WW3 spectra with track points in the L2C processor. Each strategy has its own
advantages and disadvantages in terms of storage requirements, simplicity of
implementation, and suitability for different use cases.


Summary Recommendation for Your Processor
------------------------------------------

+----------------+---------------------+---------------------+----------------------+----------------------+
| Feature        | V1 (1to1)           | V2 (Unique)         | V3 (Many)            | V4 (Lasso)           |
+================+=====================+=====================+======================+======================+
| Storage        | High (Duplicates)   | Low (Optimized)     | Medium               | Low                  |
+----------------+---------------------+---------------------+----------------------+----------------------+
| Simplicity     | Best (1:1)          | Moderate (Pointers) | Hard (Join Table)    | Simple (Footprint)   |
+----------------+---------------------+---------------------+----------------------+----------------------+
| Use Case       | Quick Visuals / ML  | Large scale Database| Scientific Validation| Subswath-level study |
+----------------+---------------------+---------------------+----------------------+----------------------+


Which one to choose?
~~~~~~~~~~~~~~~~~~~~

- If you are building a **Training Dataset for AI**, stay with V1. The redundancy
  is worth the ease of loading batches.
- If you are building a **Long-term Archive**, use V2.
- If you are investigating **Interpolation/Sub-grid variability**, use V3.
- If you want **all WW3 spectra within the SAR subswath footprint** regardless
  of tile-level matching, use V4 (lasso). This is the default mode.


Rationales Driving the L2C Processor Design Choices
----------------------------------------------------

The design choices for the L2C processor are driven by several key requirements
and trade-offs:

- Need to keep WW3 spectra with their original coordinates
- Need to keep WW3 geo-locations and times to be able to filter out associated
  spectra that are too far in space or time from the track point
- Need to separate the variables from SAR and WW3 per group


Explanation of the 4 Modes in the Script
-----------------------------------------


1to1
~~~~

**Structure:** The WW3 group has the exact same length as the SAR group.

**Behavior:** If SAR tile 10 has a match, ``ds.WW3.efth[10]`` contains the
spectrum. If no match, it contains NaN.

**Usage:** Extremely easy. No joins required. Best for machine learning.


unique
~~~~~~

**Structure:** WW3 group contains only unique spectra found for that subswath
(dimension ``unique_ww3``).

**Behavior:** ``MATCH_MAP`` contains a variable ``ww3_ptr``. To get the
spectrum for tile i, you use
``ds.WW3.efth.isel(unique_ww3=ds.MATCH_MAP.ww3_ptr[i])``.

**Usage:** Best for saving space when multiple tiles hit the same model track
point.


many
~~~~

**Structure:** Like unique, but ``MATCH_MAP`` uses a pair dimension.

**Behavior:** If one SAR tile is within the distance threshold of three
different WW3 spectra, all three pairs are recorded.

**Usage:** Best for scientific validation where you want to see all available
model points near the sensor.


lasso (Default)
~~~~~~~~~~~~~~~

**Structure:** No tile-level matching. The WW3 group contains all spectra
whose coordinates fall inside the buffered convex hull of the SAR subswath
footprint.

**Behavior:** The processor computes the convex hull of all SAR tile corner
coordinates (``oswLongitudeCorner``, ``oswLatitudeCorner``) for the subswath,
expands it by a configurable buffer (``BUFFER_DEG``, default ``0.1°``), then
selects every WW3 point within that polygon and within the temporal window.
The output contains a single ``WW3`` group (not per-tile) with the selected
spectra and their coordinates.

**Key parameters in** ``config.yml``:

- ``BUFFER_DEG``: buffer in degrees applied around the subswath convex hull
  (default ``0.1``, approximately 10 km)
- ``TIME_THRESHOLD_MINUTES``: temporal window around SAR acquisition time

**Usage:** Best when you want to study all available WW3 model information
within the SAR scene, without constraining the match to individual tiles.
Particularly suited to subswath-level spectral studies and cases where the
WW3 grid is coarser than the SAR tile grid.


Sampling the WW3 Integrated (FIELD) Parameters
----------------------------------------------

The integrated parameters (total SWH, Tm2, Tm01 and, in the future,
partitioned quantities) live in the WW3 **FIELD** files, which are 2-D gridded
fields at each hour, as opposed to the **TRACK** files that hold one spectrum
per grid cell. To attach one integrated value to every spectrum, the processor
samples the FIELD grid at each track point.

The sampling strategy is deliberately simple and unambiguous:

- **Order:** the WW3 track spectra are loaded first; the FIELD files are then
  sampled at those already-loaded track points and merged onto the same
  dataset. The integrated parameters therefore always inherit the spectra
  dimension (``sample`` in lasso mode).
- **Per-provenance-grid sampling:** every track point belongs to exactly one
  WW3 grid (recorded by ``ww3_grid_provenance``). Each point is sampled only
  from the FIELD file of **its own** grid, so a spectrum never receives values
  from the other sub-grids (Arctic / mid-latitude / Antarctic). There is exactly
  one value per spectrum per parameter — no cross-grid ambiguity.
- **Spatial matching:** the nearest FIELD grid node to the track point, found
  in the grid's native coordinates (a KD-tree over longitude/latitude with a
  date-line-aware search). A maximum distance (``ww3_field_max_distance_deg``,
  default ``0.25°``) guards against accidental matches far from the point.
- **Temporal matching:** the nearest FIELD hour within
  ``TIME_THRESHOLD_MINUTES`` of the track point time.
- **Graceful degradation:** if a grid has no FIELD file, or a point has no
  matching grid node / hour, that point keeps NaN. The product is built
  without failing, and a warning is logged.

This keeps the NetCDF output flat and easy to consume: for each WW3 spectrum,
the integrated parameters sit right next to the spectrum on the same index.

Storage Considerations
----------------------

There is not that much difference in storage between the tile-based modes
(1to1, unique, many), because in practice the WW3 grid has almost the same
posting as the SAR tile grid and we use a small radius. The lasso mode
storage depends on the subswath size and the WW3 grid resolution — it can be
significantly smaller when the WW3 grid is coarser.

Thresholds for this test:

- ``DISTANCE_THRESHOLD_KM``: 14
- ``TIME_THRESHOLD_MINUTES``: 30
- ``BUFFER_DEG``: 0.1

File sizes for test data:

- ``s1a-iw1-osw-vv-20220107t062429-20220107t062500-041351-04ea80-001_L2C_unique.nc`` — 9.7 M
- ``s1a-iw1-osw-vv-20220107t062429-20220107t062500-041351-04ea80-001_L2C_many.nc`` — 9.8 M
- ``s1a-iw1-osw-vv-20220107t062429-20220107t062500-041351-04ea80-001_L2C_1to1.nc`` — 9.7 M
