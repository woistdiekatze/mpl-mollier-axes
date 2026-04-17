from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.lines import Line2D
from matplotlib.testing.decorators import image_comparison

import mpl_mollier_axes.lines.base as lines
import mpl_mollier_axes.lines.psychrometric as plines


def setup_mollier_axes(saturation=True):
    fig = plt.figure()
    ax = fig.add_subplot(projection='mollier')
    ax.set_xlim(0, 4e-2)
    ax.set_ylim(-1e4, 5e4)
    ax.grid(visible=True, which='both')
    if saturation:
        ax.draw_saturation_line()
    return fig, ax


@pytest.fixture
def mollier_ax():
    fig, ax = setup_mollier_axes()
    yield ax
    plt.close(fig)


@pytest.fixture
def mollier_ax_no_sat():
    fig, ax = setup_mollier_axes(saturation=False)
    yield ax
    plt.close(fig)


@image_comparison(baseline_images=['sat_line'], remove_text=True, extensions=['pdf'])
def test_saturation_line(mollier_ax):
    sat = mollier_ax.draw_saturation_line(color='red')
    assert isinstance(sat, Line2D)
    assert isinstance(sat, lines.ParametricConstValueLine)
    assert isinstance(sat, plines.SaturationLine)


@image_comparison(baseline_images=['rh_lines'], remove_text=True, extensions=['pdf'])
def test_const_rh_lines(mollier_ax):
    rhs = mollier_ax.draw_const_rh_lines(*np.linspace(0.1, 0.9, 9), color='red')

    assert len(rhs) == 9
    for RHL in rhs:
        assert isinstance(RHL, Line2D)
        assert isinstance(RHL, lines.ParametricConstValueLine)
        assert isinstance(RHL, plines.ConstRhLine)


@image_comparison(baseline_images=['tdb_lines'], remove_text=True, extensions=['pdf'])
def test_const_tdb_lines(mollier_ax):
    tdbs = mollier_ax.draw_const_tdb_lines(*np.linspace(-5, 50, 12), color='red')

    assert len(tdbs) == 12
    for TDBL in tdbs:
        assert isinstance(TDBL, Line2D)
        assert isinstance(TDBL, lines.BoundedConstValueLine)
        assert isinstance(TDBL, plines.Isotherm)


@image_comparison(baseline_images=['rho_lines'], remove_text=True, extensions=['pdf'])
def test_const_rho_lines(mollier_ax):
    rhos = mollier_ax.draw_const_density_lines(*np.linspace(1.1, 1.3, 5), color='red')

    assert len(rhos) == 5
    for RHOL in rhos:
        assert isinstance(RHOL, Line2D)
        assert isinstance(RHOL, lines.ConstValueLine)
        assert isinstance(RHOL, plines.ConstDensityLine)


@image_comparison(baseline_images=['h_lines'], remove_text=True, extensions=['pdf'])
def test_const_h_lines(mollier_ax):
    hs = mollier_ax.draw_const_h_lines(*np.linspace(1e4, 3e4, 5), color='red')

    assert len(hs) == 5
    for HL in hs:
        assert isinstance(HL, Line2D)


@image_comparison(baseline_images=['w_lines'], remove_text=True, extensions=['pdf'])
def test_const_w_lines(mollier_ax):
    ws = mollier_ax.draw_const_w_lines(*np.linspace(1e-2, 3e-2, 9), color='red')

    assert len(ws) == 9
    for WL in ws:
        assert isinstance(WL, Line2D)


@pytest.mark.parametrize("pressure, baseline_images", [(p, [f'all_lines_{p:.0f}']) for p in (101325., 75000., 125000.)])
@image_comparison(baseline_images=None, remove_text=True, extensions=['pdf'])
def test_all_lines_at_different_pressures(mollier_ax_no_sat, pressure, baseline_images):
    mollier_ax_no_sat.pressure = pressure
    mollier_ax_no_sat.draw_saturation_line(color='red')
    mollier_ax_no_sat.draw_const_rh_lines(*np.linspace(0.1, 0.9, 9), color='blue')
    mollier_ax_no_sat.draw_const_tdb_lines(*np.linspace(-5, 50, 12), color='green')
    mollier_ax_no_sat.draw_const_density_lines(*np.linspace(1.1, 1.3, 5), color='grey')
    mollier_ax_no_sat.draw_const_h_lines(*np.linspace(1e4, 3e4, 5), color='magenta')
    mollier_ax_no_sat.draw_const_w_lines(*np.linspace(1e-2, 3e-2, 9), color='cyan')
