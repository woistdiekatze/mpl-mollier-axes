import mpl_mollier_axes.lines.psychrometric as plines
import numpy as np

def test_saturation_data():
    pressure = 101325.
    w, t = plines.calc_saturation_data(pressure)
    wc = np.vectorize(plines.GetSatHumRatio)(t, pressure)
    assert np.all(w == wc)
