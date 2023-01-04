from __future__ import annotations

import numpy as np
import pytest
from matplotlib.lines import Line2D

import mpl_mollier_axes as mol
import mpl_mollier_axes.lines as lines


@pytest.fixture
def ax(mollier_axes: mol.MollierAxes) -> mol.MollierAxes:
    mollier_axes.draw_saturation_line()
    return mollier_axes


@pytest.mark.fig_filename('sat_line.png')
def test_saturation_line(mollier_axes: mol.MollierAxes):
    sat = mollier_axes.draw_saturation_line(color='red')
    assert isinstance(sat, Line2D)
    assert isinstance(sat, lines.ParametricConstValueLine)


@pytest.mark.fig_filename('rh_lines.png')
def test_const_rh_lines(ax: mol.MollierAxes):
    rhs = ax.draw_const_rh_lines(*np.linspace(0.1, 0.9, 9), color='red')

    assert len(rhs) == 9
    for RHL in rhs:
        assert isinstance(RHL, Line2D)
        assert isinstance(RHL, lines.ParametricConstValueLine)


@pytest.mark.fig_filename('tdb_lines.png')
def test_const_tdb_lines(ax: mol.MollierAxes):
    tdbs = ax.draw_const_tdb_lines(*np.linspace(-5, 50, 12), color='red')

    assert len(tdbs) == 12
    for TDBL in tdbs:
        assert isinstance(TDBL, Line2D)
        assert isinstance(TDBL, lines.BoundedConstValueLine)


@pytest.mark.fig_filename('rho_lines.png')
def test_const_rho_lines(ax: mol.MollierAxes):
    rhos = ax.draw_const_density_lines(*np.linspace(1.1, 1.3, 5), color='red')

    assert len(rhos) == 5
    for RHOL in rhos:
        assert isinstance(RHOL, Line2D)
        assert isinstance(RHOL, lines.ConstValueLine)


@pytest.mark.fig_filename('h_lines.png')
def test_const_h_lines(ax: mol.MollierAxes):
    hs = ax.draw_const_h_lines(*np.linspace(1e4, 3e4, 5), color='red')

    assert len(hs) == 5
    for HL in hs:
        assert isinstance(HL, Line2D)


@pytest.mark.fig_filename('w_lines.png')
def test_const_w_lines(ax: mol.MollierAxes):
    ws = ax.draw_const_w_lines(*np.linspace(1e-2, 3e-2, 9), color='red')

    assert len(ws) == 9
    for WL in ws:
        assert isinstance(WL, Line2D)
