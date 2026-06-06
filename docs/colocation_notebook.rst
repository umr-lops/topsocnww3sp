.. _colocation_notebook:

Co-location Illustration (Notebook)
===================================

The co-localization process allows for the alignment of SAR and WW3 data. This page describes how to implement a workflow using a notebook.

Typical Notebook Workflow
--------------------------

A typical analysis notebook would follow these steps:

1. **Data Loading**: Load an L2C NetCDF file produced by ``procl2c``.
2. **Visualization**: Plot the SAR footprint and the associated WW3 points.
3. **Spectral Analysis**: Extract spectral parameters (e.g., significant wave height) from both the OSW product and the WW3 spectra for the same coordinates.
4. **Validation**: Compare values using scatter plots or time-series analysis.

Example snippet for visualization:

.. code-block:: python

    import matplotlib.pyplot as plt
    import xarray as xr
    import cartopy.crs as ccrs

    # Load data
    ds_sar = xr.open_dataset("product_v0.1.nc", group="SAR_intraburst")
    ds_ww3 = xr.open_dataset("product_v0.1.nc", group="WW3")

    # Plotting map
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # The SAR tiles can be plotted as points or polygons
    ax.scatter(ds_sar.oswLon, ds_sar.oswLat, c='blue', label='SAR Tiles')
    ax.scatter(ds_ww3.longitude, ds_ww3.latitude, c='red', label='WW3 Spectra')
    ax.legend()
    plt.show()
