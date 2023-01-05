from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.lines import Line2D
from matplotlib.testing.decorators import image_comparison

import mpl_mollier_axes.lines as lines


def setup_mollier_axes(saturation=True):
    fig = plt.figure()
    ax = fig.add_subplot(projection='mollier')
    ax.set_xlim(0, 4e-2)
    ax.set_ylim(-1e4, 5e4)
    ax.grid(visible=True, which='both')
    if saturation:
        ax.draw_saturation_line()
    return fig, ax


@image_comparison(baseline_images=['sat_line'],
                  remove_text=True,
                  extensions=['pdf'])
def test_saturation_line():
    fig, ax = setup_mollier_axes(False)
    sat = ax.draw_saturation_line(color='red')
    assert isinstance(sat, Line2D)
    assert isinstance(sat, lines.ParametricConstValueLine)


@image_comparison(baseline_images=['rh_lines'],
                  remove_text=True,
                  extensions=['pdf'])
def test_const_rh_lines():
    fig, ax = setup_mollier_axes()
    rhs = ax.draw_const_rh_lines(*np.linspace(0.1, 0.9, 9), color='red')

    assert len(rhs) == 9
    for RHL in rhs:
        assert isinstance(RHL, Line2D)
        assert isinstance(RHL, lines.ParametricConstValueLine)


@image_comparison(baseline_images=['tdb_lines'],
                  remove_text=True,
                  extensions=['pdf'])
def test_const_tdb_lines():
    fig, ax = setup_mollier_axes()
    tdbs = ax.draw_const_tdb_lines(*np.linspace(-5, 50, 12), color='red')

    assert len(tdbs) == 12
    for TDBL in tdbs:
        assert isinstance(TDBL, Line2D)
        assert isinstance(TDBL, lines.BoundedConstValueLine)


@image_comparison(baseline_images=['rho_lines'],
                  remove_text=True,
                  extensions=['pdf'])
def test_const_rho_lines():
    fig, ax = setup_mollier_axes()
    rhos = ax.draw_const_density_lines(*np.linspace(1.1, 1.3, 5), color='red')

    assert len(rhos) == 5
    for RHOL in rhos:
        assert isinstance(RHOL, Line2D)
        assert isinstance(RHOL, lines.ConstValueLine)


@image_comparison(baseline_images=['h_lines'],
                  remove_text=True,
                  extensions=['pdf'])
def test_const_h_lines():
    fig, ax = setup_mollier_axes()
    hs = ax.draw_const_h_lines(*np.linspace(1e4, 3e4, 5), color='red')

    assert len(hs) == 5
    for HL in hs:
        assert isinstance(HL, Line2D)


@image_comparison(baseline_images=['w_lines'],
                  remove_text=True,
                  extensions=['pdf'])
def test_const_w_lines():
    fig, ax = setup_mollier_axes()
    ws = ax.draw_const_w_lines(*np.linspace(1e-2, 3e-2, 9), color='red')

    assert len(ws) == 9
    for WL in ws:
        assert isinstance(WL, Line2D)


@pytest.mark.parametrize("pressure, baseline_images",
                         [(p, [f'all_lines_{p:.0f}'])
                          for p in (101325., 75000., 125000.)])
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
