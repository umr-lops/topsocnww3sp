Exploration of Different Strategies for the L2C Processor
============================================================

This document explores different strategies for the L2C processor to determine which WW3 spectra to associate with each track point, and how to handle cases where no spectra are found within the defined spatiotemporal thresholds. This includes testing different distance and time thresholds, as well as fallback strategies such as using the nearest spectrum or interpolating between nearby spectra.

During development, we have explored three different strategies for associating WW3 spectra with track points in the L2C processor. Each strategy has its own advantages and disadvantages in terms of storage requirements, simplicity of implementation, and suitability for different use cases.

## Summary Recommendation for Your Processor

+----------------+---------------------+---------------------+----------------------+
| Feature        | V1 (Current)        | V2 (Unique)         | V3 (One-to-Many)     |
+================+=====================+=====================+======================+
| Storage        | High (Duplicates)   | Low (Optimized)     | Medium               |
+----------------+---------------------+---------------------+----------------------+
| Simplicity     | Best (1:1)          | Moderate (Pointers) | Hard (Join Table)    |
+----------------+---------------------+---------------------+----------------------+
| Use Case       | Quick Visuals / ML  | Large scale Database| Scientific Validation|
+----------------+---------------------+---------------------+----------------------+

### Which one to choose?

- If you are building a Training Dataset for AI, stay with V1. The redundancy is worth the ease of loading batches.
- If you are building a Long-term Archive, use V2.
- If you are investigating Interpolation/Sub-grid variability, use V3.

## Rationales Driving the L2C Processor Design Choices

The design choices for the L2C processor are driven by several key requirements and trade-offs:

- Need to keep WW3 spectra with their original coordinates
- Need to keep WW3 geo-locations and times to be able to filter out associated spectra that are too far in space or time from the track point
- Need to separate the variables from SAR and WW3 per group

## Explanation of the 3 Modes in the Script

### 1to1 (Default)

**Structure:** The WW3 group has the exact same length as the SAR group.

**Behavior:** If SAR tile 10 has a match, ``ds.WW3.efth[10]`` contains the spectrum. If no match, it contains NaN.

**Usage:** Extremely easy. No joins required. Best for machine learning.

### unique

**Structure:** WW3 group contains only unique spectra found for that subswath (dimension ``unique_ww3``).

**Behavior:** ``MATCH_MAP`` contains a variable ``ww3_ptr``. To get the spectrum for tile i, you use ``ds.WW3.efth.isel(unique_ww3=ds.MATCH_MAP.ww3_ptr[i])``.

**Usage:** Best for saving space when multiple tiles hit the same model track point.

### many

**Structure:** Like unique, but ``MATCH_MAP`` uses a pair dimension.

**Behavior:** If one SAR tile is within 20km of three different WW3 spectra, all three pairs are recorded.

**Usage:** Best for scientific validation where you want to see all available model points near the sensor.

## Storage Considerations

There is no that much difference in storage between the three modes, because in practice we have a WW3 grid that has almost same posting than the SAR grid and we small radius.

**Thresholds for this test:**
- DISTANCE_THRESHOLD_KM: 14
- TIME_THRESHOLD_MINUTES: 30

**File sizes for test data:**
- ``s1a-iw1-osw-vv-20220107t062429-20220107t062500-041351-04ea80-001_L2C_unique.nc*`` - 9.7M
- ``s1a-iw1-osw-vv-20220107t062429-20220107t062500-041351-04ea80-001_L2C_many.nc*`` - 9.8M
- ``s1a-iw1-osw-vv-20220107t062429-20220107t062500-041351-04ea80-001_L2C_1to1.nc*`` - 9.7M
