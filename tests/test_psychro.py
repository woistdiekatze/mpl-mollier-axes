import numpy as np

import mpl_mollier_axes.lines.psychrometric as plines
from mpl_mollier_axes.backends import get_default_backend


def test_saturation_data():
    pressure = 101325.
    backend = get_default_backend()
    w, t = plines.calc_saturation_data(pressure)
    wc = np.vectorize(backend.get_sat_hum_ratio)(t, pressure)
    assert np.all(w == wc)
