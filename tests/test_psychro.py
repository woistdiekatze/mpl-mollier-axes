import numpy as np

import mpl_mollier_axes.lines.psychrometric as plines


def test_saturation_data():
    pressure = 101325.
    w, t = plines.calc_saturation_data(pressure)
    wc = np.vectorize(plines.GetSatHumRatio)(t, pressure)
    assert np.all(w == wc)
