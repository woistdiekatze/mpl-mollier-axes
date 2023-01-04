from math import atan

from mpl_mollier_axes import MollierAxes, MollierProjection
from mpl_mollier_axes.axes import SkewYAxes, SkewYProjection
from mpl_mollier_axes.axes.MollierAxes import _H_EVAP_H2O_0CELSIUS


def near(a, b, reltol=1e-9):
    return 2 * abs(a - b) / (a + b) <= reltol


def test_skew_y_axes(fig):
    angle = 20
    ax = fig.add_subplot(projection=SkewYProjection(skewy_deg=angle))
    assert isinstance(ax, SkewYAxes)
    a, b, c, d, e, f = ax.transAffine.to_values()
    assert a == 1
    assert near(b, -atan(angle))
    assert c == 0
    assert d == 1
    assert e == 0
    assert f == 0


def test_mollier_axes(fig):
    ax = fig.add_subplot(projection='mollier')
    assert isinstance(ax, MollierAxes)
    a, b, c, d, e, f = ax.transAffine.to_values()
    assert a == 1
    assert near(b, -atan(_H_EVAP_H2O_0CELSIUS))
    assert c == 0
    assert d == 1
    assert e == 0
    assert f == 0


def test_mollier_projection(fig):
    ax = fig.add_subplot(projection=MollierProjection(pressure=750000))
    assert isinstance(ax, MollierAxes)
    a, b, c, d, e, f = ax.transAffine.to_values()
    assert a == 1
    assert near(b, -atan(_H_EVAP_H2O_0CELSIUS))
    assert c == 0
    assert d == 1
    assert e == 0
    assert f == 0
