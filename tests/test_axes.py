from math import atan

import pytest
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison

from mpl_mollier_axes import MollierAxes, MollierProjection
from mpl_mollier_axes.axes import SkewYAxes, SkewYProjection
from mpl_mollier_axes.axes.MollierAxes import _H_EVAP_H2O_0CELSIUS


def near(a, b, reltol=1e-9):
    return 2 * abs(a - b) / (a + b) <= reltol


@image_comparison(baseline_images=['skew_y_proj'],
                  remove_text=True,
                  extensions=['pdf'])
def test_skew_y_axes():
    angle = 20
    fig = plt.figure()
    ax = fig.add_subplot(projection=SkewYProjection(skewy_deg=angle))
    ax.grid(True, which='both')
    assert isinstance(ax, SkewYAxes)
    a, b, c, d, e, f = ax.transAffine.to_values()
    assert a == 1
    assert near(b, -atan(angle))
    assert c == 0
    assert d == 1
    assert e == 0
    assert f == 0


def setup_mollier_axes(ax):
    ax.set_xlim(0, 1e-3)
    ax.set_ylim(0, 1e4)
    ax.set_aspect(1e-7)
    ax.grid(True, which='both')


@image_comparison(baseline_images=['mollier_axes'],
                  remove_text=True,
                  extensions=['pdf'])
def test_mollier_axes():
    fig = plt.figure()
    ax = fig.add_subplot(projection='mollier')
    setup_mollier_axes(ax)

    assert isinstance(ax, MollierAxes)
    a, b, c, d, e, f = ax.transAffine.to_values()
    assert a == 1
    assert near(b, -atan(_H_EVAP_H2O_0CELSIUS))
    assert c == 0
    assert d == 1
    assert e == 0
    assert f == 0


@check_figures_equal(extensions=['pdf'])
def test_mollier_projection(fig_test, fig_ref):
    ax_ref = fig_ref.add_subplot(projection='mollier')
    setup_mollier_axes(ax_ref)

    ax_test = fig_test.add_subplot(projection=MollierProjection())
    setup_mollier_axes(ax_test)

    assert isinstance(ax_test, MollierAxes)
    a, b, c, d, e, f = ax_test.transAffine.to_values()
    assert a == 1
    assert near(b, -atan(_H_EVAP_H2O_0CELSIUS))
    assert c == 0
    assert d == 1
    assert e == 0
    assert f == 0


@check_figures_equal(extensions=['pdf'])
def test_mollier_projection_different_pressures(fig_test, fig_ref):
    ax_ref = fig_ref.add_subplot(projection=MollierProjection())
    setup_mollier_axes(ax_ref)

    ax_test = fig_test.add_subplot(projection=MollierProjection(
        pressure=750000))
    setup_mollier_axes(ax_test)


def test_pressure_gt_0():
    fig = plt.figure()
    ax = fig.add_subplot(projection='mollier')
    with pytest.raises(ValueError, match="pressure must be greater than zero."):
        ax.pressure = -1.
