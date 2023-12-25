"""Test functionality of pyxlma.lmalib.flash"""

import xarray as xr
import numpy as np
from pyxlma.lmalib.flash.cluster import cluster_flashes
from pyxlma.lmalib.flash.properties import flash_stats, filter_flashes


def test_cluster_flashes():
    """Test clustering of flashes"""
    dataset = xr.open_dataset('examples/data/lma_netcdf/lma.nc')
    clustered = cluster_flashes(dataset)
    truth = xr.open_dataset('examples/data/lma_netcdf/lma_clustered.nc')


    for var in truth.data_vars:
        if var == 'flash_time_separation_threshold':
            truth[var].data = truth[var].data.astype(float)/1e9
        np.testing.assert_allclose(clustered[var].data, truth[var].data)


def test_flash_stats():
    """Test calculation of flash statistics"""
    dataset = xr.open_dataset('examples/data/lma_netcdf/lma.nc')
    stats = flash_stats(cluster_flashes(dataset))
    truth = xr.open_dataset('examples/data/lma_netcdf/lma_stats.nc')
    for var in truth.data_vars:
        if var == 'flash_time_separation_threshold':
            truth[var].data = truth[var].data.astype(float)/1e9
        elif truth[var].data.dtype == 'datetime64[ns]':
            np.testing.assert_allclose(stats[var].data.astype(float), truth[var].data.astype(float))
        else:
            np.testing.assert_allclose(stats[var].data, truth[var].data)
        