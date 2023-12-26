import pytest
import xarray as xr
from pyxlma.plot.xlma_base_plot import *
from pyxlma.plot.xlma_plot_feature import *
import datetime as dt

@pytest.mark.mpl_image_compare
def test_blank_plot():
    start_time = dt.datetime(2023, 12, 24, 0, 57, 0, 0)
    end_time = start_time + dt.timedelta(seconds=60)
    bk_plot = BlankPlot(start_time, bkgmap=True, xlim=[-103.5, -99.5], ylim=[31.5, 35.5], zlim=[0, 20], tlim=[start_time, end_time], title='XLMA Test Plot')
    return bk_plot.fig

@pytest.mark.mpl_image_compare
def test_blank_plot_labeled():
    start_time = dt.datetime(2023, 12, 24, 0, 57, 0, 0)
    end_time = start_time + dt.timedelta(seconds=60)
    bk_plot = BlankPlot(start_time, bkgmap=True, xlim=[-103.5, -99.5], ylim=[31.5, 35.5], zlim=[0, 20], tlim=[start_time, end_time], title='XLMA Test Plot')
    subplot_labels(bk_plot)
    return bk_plot.fig
