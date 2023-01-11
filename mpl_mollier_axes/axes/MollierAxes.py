from math import atan

from psychrolib import SI, GetStandardAtmPressure, SetUnitSystem
from matplotlib.projections import register_projection

from ..lines.psychrometric import ConstDensityLine, ConstRhLine, Isotherm, SaturationLine
from .SkewYAxes import SkewYAxes

SetUnitSystem(SI)
_H_EVAP_H2O_0CELSIUS = 2.501e6  # J / kg


class MollierAxes(SkewYAxes):
    name = 'mollier'

    def __init__(self, *args, pressure: float = GetStandardAtmPressure(0.), **kwargs) -> None:
        # The skew transformation will be carried out in data coordinates.
        # Thus the skew angle corresponds to the heat of evaporation of pure water at 0 dgrees celsius.
        super().__init__(*args, skewy=-atan(_H_EVAP_H2O_0CELSIUS), **kwargs)
        self.pressure = pressure

    @property
    def pressure(self):
        return self._pressure

    @pressure.setter
    def pressure(self, pressure: float):
        if pressure <= 0:
            raise ValueError("pressure must be greater than zero.")
        self._pressure = pressure

    def draw_const_h_lines(self, *h, color='black', linewidth=.5, **kwargs):
        return [self.axhline(H, color=color, linewidth=linewidth, **kwargs) for H in h]

    def draw_const_w_lines(self, *w, color='black', linewidth=.5, **kwargs):
        return [self.axvline(W, color=color, linewidth=linewidth, **kwargs) for W in w]

    def draw_const_rh_lines(self, *rh, color='black', linewidth=.5, **kwargs):
        return [self.add_line(ConstRhLine(RH, self.pressure, color=color, linewidth=linewidth, **kwargs)) for RH in rh]

    def draw_const_tdb_lines(self, *tdb, color='black', linewidth=.5, **kwargs):
        return [self.add_line(Isotherm(TDB, self.pressure, color=color, linewidth=linewidth, **kwargs)) for TDB in tdb]

    def draw_const_density_lines(self, *rho, color='black', linewidth=.5, **kwargs):
        return [
            self.add_line(ConstDensityLine(RHO, self.pressure, color=color, linewidth=linewidth, **kwargs))
            for RHO in rho
        ]

    def draw_saturation_line(self, color='black', linewidth=2, **kwargs):
        return self.add_line(SaturationLine(self.pressure, color=color, linewidth=linewidth, **kwargs))


class MollierProjection:

    def __init__(self, pressure: float = GetStandardAtmPressure(0.)) -> None:
        self._pressure = pressure

    def _as_mpl_axes(self):
        return MollierAxes, dict(pressure=self._pressure)


register_projection(MollierAxes)
