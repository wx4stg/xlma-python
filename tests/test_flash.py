"""Test functionality of pyxlma.lmalib.flash"""

import xarray as xr
import numpy as np
from pyxlma.lmalib.flash.cluster import cluster_flashes
from pyxlma.lmalib.flash.properties import flash_stats, filter_flashes



def compare_dataarrays(tocheck, truth, var):
    """Compare two dataarrays"""
    if truth[var].data.dtype == 'datetime64[ns]' or truth[var].data.dtype == 'timedelta64[ns]':
        if tocheck[var].data.dtype == 'float64':
            truth[var].data = truth[var].data.astype(float)/1e9
        np.testing.assert_allclose(tocheck[var].data.astype(float), truth[var].data.astype(float))
    else:
        np.testing.assert_allclose(tocheck[var].data, truth[var].data)


def test_cluster_flashes():
    """Test clustering of flashes"""
    dataset = xr.open_dataset('examples/data/lma_netcdf/lma.nc')
    clustered = cluster_flashes(dataset)
    truth = xr.open_dataset('examples/data/lma_netcdf/lma_clustered.nc')
    for var in truth.data_vars:
        compare_dataarrays(clustered, truth, var)


def test_flash_stats():
    """Test calculation of flash statistics"""
    dataset = xr.open_dataset('examples/data/lma_netcdf/lma.nc')
    stats = flash_stats(cluster_flashes(dataset))
    truth = xr.open_dataset('examples/data/lma_netcdf/lma_stats.nc')
    for var in truth.data_vars:
        compare_dataarrays(stats, truth, var)


def test_filter_flashes():
    """Test filtering of flashes"""
    dataset = xr.open_dataset('examples/data/lma_netcdf/lma_stats.nc')
    filtered = filter_flashes(dataset, flash_event_count=(100, None))
    assert np.min(filtered.flash_event_count.data) >= 100
    truth = xr.open_dataset('examples/data/lma_netcdf/lma_filtered.nc')

    for var in truth.data_vars:
        compare_dataarrays(filtered, truth, var)