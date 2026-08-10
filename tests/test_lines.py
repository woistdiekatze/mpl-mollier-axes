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


@image_comparison(baseline_images=['sat_line'], remove_text=True, extensions=['pdf'])
def test_saturation_line():
    fig, ax = setup_mollier_axes(False)
    sat = ax.draw_saturation_line(color='red')
    assert isinstance(sat, Line2D)
    assert isinstance(sat, lines.ParametricConstValueLine)
    assert isinstance(sat, plines.SaturationLine)


@image_comparison(baseline_images=['rh_lines'], remove_text=True, extensions=['pdf'])
def test_const_rh_lines():
    fig, ax = setup_mollier_axes()
    rhs = ax.draw_const_rh_lines(*np.linspace(0.1, 0.9, 9), color='red')

    assert len(rhs) == 9
    for RHL in rhs:
        assert isinstance(RHL, Line2D)
        assert isinstance(RHL, lines.ParametricConstValueLine)
        assert isinstance(RHL, plines.ConstRhLine)


@image_comparison(baseline_images=['tdb_lines'], remove_text=True, extensions=['pdf'])
def test_const_tdb_lines():
    fig, ax = setup_mollier_axes()
    tdbs = ax.draw_const_tdb_lines(*np.linspace(-5, 50, 12), color='red')

    assert len(tdbs) == 12
    for TDBL in tdbs:
        assert isinstance(TDBL, Line2D)
        assert isinstance(TDBL, lines.BoundedConstValueLine)
        assert isinstance(TDBL, plines.Isotherm)


@image_comparison(baseline_images=['rho_lines'], remove_text=True, extensions=['pdf'])
def test_const_rho_lines():
    fig, ax = setup_mollier_axes()
    rhos = ax.draw_const_density_lines(*np.linspace(1.1, 1.3, 5), color='red')

    assert len(rhos) == 5
    for RHOL in rhos:
        assert isinstance(RHOL, Line2D)
        assert isinstance(RHOL, lines.ConstValueLine)
        assert isinstance(RHOL, plines.ConstDensityLine)


@image_comparison(baseline_images=['h_lines'], remove_text=True, extensions=['pdf'])
def test_const_h_lines():
    fig, ax = setup_mollier_axes()
    hs = ax.draw_const_h_lines(*np.linspace(1e4, 3e4, 5), color='red')

    assert len(hs) == 5
    for HL in hs:
        assert isinstance(HL, Line2D)


@image_comparison(baseline_images=['w_lines'], remove_text=True, extensions=['pdf'])
def test_const_w_lines():
    fig, ax = setup_mollier_axes()
    ws = ax.draw_const_w_lines(*np.linspace(1e-2, 3e-2, 9), color='red')

    assert len(ws) == 9
    for WL in ws:
        assert isinstance(WL, Line2D)


@pytest.mark.parametrize("pressure, baseline_images", [(p, [f'all_lines_{p:.0f}']) for p in (101325., 75000., 125000.)])
@image_comparison(baseline_images=None, remove_text=True, extensions=['pdf'])
def test_all_lines_at_different_pressures(pressure, baseline_images):
    fig, ax = setup_mollier_axes(False)
    ax.pressure = pressure
    ax.draw_saturation_line(color='red')
    ax.draw_const_rh_lines(*np.linspace(0.1, 0.9, 9), color='blue')
    ax.draw_const_tdb_lines(*np.linspace(-5, 50, 12), color='green')
    ax.draw_const_density_lines(*np.linspace(1.1, 1.3, 5), color='grey')
    ax.draw_const_h_lines(*np.linspace(1e4, 3e4, 5), color='magenta')
    ax.draw_const_w_lines(*np.linspace(1e-2, 3e-2, 9), color='cyan')


def test_const_rh_lines_track_pressure_changes():
    fig, ax = setup_mollier_axes(False)
    rh, = ax.draw_const_rh_lines(0.5)
    fig.canvas.draw()
    data_at_atm = rh.get_data()

    ax.pressure = 75000.
    fig.canvas.draw()
    data_at_low_pressure = rh.get_data()

    assert not np.array_equal(data_at_atm, data_at_low_pressure)


def test_saturation_line_tracks_pressure_changes():
    fig, ax = setup_mollier_axes(False)
    sat = ax.draw_saturation_line()
    fig.canvas.draw()
    data_at_atm = sat.get_data()

    ax.pressure = 75000.
    fig.canvas.draw()
    data_at_low_pressure = sat.get_data()

    assert not np.array_equal(data_at_atm, data_at_low_pressure)


def test_grid_lines_default_below_data_zorder():
    fig, ax = setup_mollier_axes()
    rh, = ax.draw_const_rh_lines(0.5)
    tdb, = ax.draw_const_tdb_lines(20.)
    rho, = ax.draw_const_density_lines(1.2)
    h, = ax.draw_const_h_lines(2e4)
    w, = ax.draw_const_w_lines(2e-2)
    sat = ax.draw_saturation_line()

    default_line_zorder = Line2D([], []).get_zorder()
    for grid_line in (rh, tdb, rho, h, w):
        assert grid_line.get_zorder() < default_line_zorder
    # The saturation boundary is drawn above the other backdrop lines, but
    # still below regular plot data.
    assert rh.get_zorder() < sat.get_zorder() < default_line_zorder
