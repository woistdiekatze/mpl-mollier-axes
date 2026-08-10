from math import atan

from matplotlib.projections import register_projection
from psychrolib import SI, GetStandardAtmPressure, SetUnitSystem

from ..backends import PsychrometricBackend, get_default_backend
from ..lines.psychrometric import (ConstDensityLine, ConstRhLine, Isotherm,
                                   SaturationLine)
from .SkewYAxes import SkewYAxes

SetUnitSystem(SI)
_H_EVAP_H2O_0CELSIUS = 2.501e6  # J / kg

# Default z-order for the "backdrop" lines of constant property value, kept
# below the default Line2D/scatter z-order (2) so that data plotted onto the
# chart afterwards sits on top of it, similar to how axis gridlines are drawn
# below regular plot content by default. The saturation line is drawn above
# the other backdrop lines (but still below regular plot data) since it is a
# hard physical boundary rather than just a reference gridline.
_GRID_ZORDER = 1.
_SATURATION_ZORDER = 1.5


class MollierAxes(SkewYAxes):
    name = 'mollier'

    def __init__(
        self,
        *args,
        pressure: float = GetStandardAtmPressure(0.),
        backend: PsychrometricBackend = None,
        **kwargs
    ) -> None:
        # The skew transformation will be carried out in data coordinates.
        # Thus the skew angle corresponds to the heat of evaporation of pure water at 0 dgrees celsius.
        super().__init__(*args, skewy=-atan(_H_EVAP_H2O_0CELSIUS), **kwargs)
        self.pressure = pressure
        self._backend = backend or get_default_backend()

    @property
    def pressure(self):
        return self._pressure

    @pressure.setter
    def pressure(self, pressure: float):
        if pressure <= 0:
            raise ValueError("pressure must be greater than zero.")
        self._pressure = pressure

    def draw_const_h_lines(self, *h, color='black', linewidth=.5, zorder=_GRID_ZORDER, **kwargs):
        return [self.axhline(H, color=color, linewidth=linewidth, zorder=zorder, **kwargs) for H in h]

    def draw_const_w_lines(self, *w, color='black', linewidth=.5, zorder=_GRID_ZORDER, **kwargs):
        return [self.axvline(W, color=color, linewidth=linewidth, zorder=zorder, **kwargs) for W in w]

    def draw_const_rh_lines(self, *rh, color='black', linewidth=.5, zorder=_GRID_ZORDER, **kwargs):
        # Pressure is read live from `self.pressure` on every recalculation (see
        # ConstRhLine), so lines drawn here keep tracking it if it is changed later.
        line_kwargs = dict(backend=self._backend, color=color, linewidth=linewidth, zorder=zorder, **kwargs)
        return [self.add_line(ConstRhLine(RH, **line_kwargs)) for RH in rh]

    def draw_const_tdb_lines(self, *tdb, color='black', linewidth=.5, zorder=_GRID_ZORDER, **kwargs):
        line_kwargs = dict(backend=self._backend, color=color, linewidth=linewidth, zorder=zorder, **kwargs)
        return [self.add_line(Isotherm(TDB, self.pressure, **line_kwargs)) for TDB in tdb]

    def draw_const_density_lines(self, *rho, color='black', linewidth=.5, zorder=_GRID_ZORDER, **kwargs):
        line_kwargs = dict(backend=self._backend, color=color, linewidth=linewidth, zorder=zorder, **kwargs)
        return [self.add_line(ConstDensityLine(RHO, self.pressure, **line_kwargs)) for RHO in rho]

    def draw_saturation_line(self, color='black', linewidth=2, zorder=_SATURATION_ZORDER, **kwargs):
        # Pressure is read live from `self.pressure` on every recalculation (see
        # SaturationLine), so this line keeps tracking it if it is changed later.
        line_kwargs = dict(backend=self._backend, color=color, linewidth=linewidth, zorder=zorder, **kwargs)
        return self.add_line(SaturationLine(**line_kwargs))


class MollierProjection:

    def __init__(self, pressure: float = GetStandardAtmPressure(0.), backend: PsychrometricBackend = None) -> None:
        self._pressure = pressure
        self._backend = backend

    def _as_mpl_axes(self):
        return MollierAxes, dict(pressure=self._pressure, backend=self._backend)


register_projection(MollierAxes)
