from math import atan

import matplotlib.pyplot as plt

from mpl_mollier_axes import MollierAxes, __version__
from mpl_mollier_axes.axes import SkewYAxes, SkewYProjection


def test_version():
    assert __version__ == '0.1.0'


def near(a, b, reltol=1e-9):
    return 2 * abs(a - b) / (a + b) <= reltol


def test_skew_y_axes():
    angle = 20
    fig = plt.figure()
    ax = fig.add_subplot(projection=SkewYProjection(skewy_deg=angle))
    assert isinstance(ax, SkewYAxes)
    a, b, c, d, e, f = ax.transAffine.to_values()
    assert a == 1
    assert near(b, -atan(angle))
    assert c == 0
    assert d == 1
    assert e == 0
    assert f == 0


def test_mollier_axes():
    fig = plt.figure()
    ax = fig.add_subplot(projection='mollier')
    assert isinstance(ax, MollierAxes)
    a, b, c, d, e, f = ax.transAffine.to_values()
    assert a == 1
    assert near(b, -atan(2.501e6))
    assert c == 0
    assert d == 1
    assert e == 0
    assert f == 0
