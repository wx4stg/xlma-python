from pyxlma.coords import *
import pytest
import numpy as np

test_lats = np.array([33.5, 1.0, 0.0, 0.0, 0.0, 10.0, -10.0, 33.606968])
test_lons = np.array([-101.5, -75.0, -85.0, -65.0, -75.0, -75.0, -75.0, -101.822625])
test_alts = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 984.0])

test_ecef_X = np.array([-1061448.75418035, 1650533.58831094, 555891.26758132,
                        2695517.17208404, 1650783.32787306, 1625868.32721344,
                        1625868.32721344, -1089633.44245767])
test_ecef_Y = np.array([-5217187.30723133, -6159875.21117539, -6353866.26310279,
                        -5780555.22988658, -6160807.25190988, -6067823.20357756,
                        -6067823.20357756, -5205511.43302535])
test_ecef_Z = np.array([3500334.28802236, 110568.77482457, 0, 0,
                        0, 1100248.54773536, -1100248.54773536, 3510766.26631805])

def test_geographic():
    geosys = GeographicSystem()
    ecef_coords = geosys.toECEF(test_lons, test_lats, test_alts)
    lons, lats, alts = geosys.fromECEF(*ecef_coords)

    assert np.allclose(ecef_coords[0], test_ecef_X)
    assert np.allclose(ecef_coords[1], test_ecef_Y)
    assert np.allclose(ecef_coords[2], test_ecef_Z)
    assert np.allclose(lons, test_lons)
    assert np.allclose(lats, test_lats)
    assert np.allclose(alts, test_alts)

def test_geographic_one_point():
    geosys = GeographicSystem()
    ecef_coords = geosys.toECEF(test_lons[-1], test_lats[-1], test_alts[-1])
    lons, lats, alts = geosys.fromECEF(*ecef_coords)

    assert np.allclose(ecef_coords[0], test_ecef_X[-1])
    assert np.allclose(ecef_coords[1], test_ecef_Y[-1])
    assert np.allclose(ecef_coords[2], test_ecef_Z[-1])
    assert np.allclose(lons[0], test_lons[-1])
    assert np.allclose(lats[0], test_lats[-1])
    assert np.allclose(alts[0], test_alts[-1])

def test_geographic_custom_r_both():
    geosys = GeographicSystem(r_equator=6378.137, r_pole=6356.752)
    ecef_coords = geosys.toECEF(test_lons, test_lats, test_alts)
    lons, lats, alts = geosys.fromECEF(*ecef_coords)
    assert np.allclose(lons, test_lons)
    assert np.allclose(lats, test_lats)
    assert np.allclose(alts, test_alts)

def test_geographic_custom_r_eq():
    geosys = GeographicSystem(r_equator=6378.137)
    ecef_coords = geosys.toECEF(test_lons, test_lats, test_alts)
    lons, lats, alts = geosys.fromECEF(*ecef_coords)
    assert np.allclose(lons, test_lons)
    assert np.allclose(lats, test_lats)
    assert np.allclose(alts, test_alts)

def test_geographic_custom_r_pole():
    geosys = GeographicSystem(r_pole=6356.752)
    ecef_coords = geosys.toECEF(test_lons, test_lats, test_alts)
    lons, lats, alts = geosys.fromECEF(*ecef_coords)
    assert np.allclose(lons, test_lons)
    assert np.allclose(lats, test_lats)
    assert np.allclose(alts, test_alts)

def test_equidistant_cylindrical():
    eqsys = MapProjection(ctrLat=test_lats[-1], ctrLon=test_lons[-1])
    ecef_coords = eqsys.toECEF(0, 0, 0)
    x, y, z = eqsys.fromECEF(*ecef_coords)
    assert np.allclose(x, 0)
    assert np.allclose(y, 0)
    assert np.allclose(z, 0)

