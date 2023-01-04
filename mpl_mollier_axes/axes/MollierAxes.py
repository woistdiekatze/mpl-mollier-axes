from math import atan

import numpy as np
from matplotlib.projections import register_projection
from psychrolib import (SI, GetHumRatioFromRelHum, GetMoistAirEnthalpy,
                        GetSatAirEnthalpy, GetSatHumRatio,
                        GetStandardAtmPressure,
                        GetTDryBulbFromMoistAirVolumeAndHumRatio,
                        SetUnitSystem)

from ..lines import (BoundedConstValueLine, ConstValueLine,
                     ParametricConstValueLine)
from .SkewYAxes import SkewYAxes

SetUnitSystem(SI)
_H_EVAP_H2O_0CELSIUS = 2.501e6  # J / kg
_CP_DRY_AIR = 1.006e3  # J / kg K


class MollierAxes(SkewYAxes):
    name = 'mollier'

    def __init__(self,
                 *args,
                 pressure: float = GetStandardAtmPressure(0.),
                 **kwargs) -> None:
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

        def _draw_const_rh_line(rh):

            def bound_fun():
                hmin, hmax = self.get_ybound()
                tmin = hmin / _CP_DRY_AIR
                tmax = hmax / _CP_DRY_AIR
                return tmin, tmax

            def calc_fun(t, rh):
                w = GetHumRatioFromRelHum(t, rh, self.pressure)
                h = GetMoistAirEnthalpy(t, w)
                return w, h

            line = ParametricConstValueLine(rh, calc_fun, bound_fun, color=color, linewidth=linewidth, **kwargs)
            return self.add_line(line)

        return [_draw_const_rh_line(RH) for RH in rh]

    def draw_const_tdb_lines(self, *tdb, color='black', linewidth=.5, **kwargs):

        def _draw_const_tdb_line(tdb):
            wsat = GetSatHumRatio(tdb, self.pressure)
            line = BoundedConstValueLine(
                tdb,
                lambda w, t: GetMoistAirEnthalpy(t, w),
                0,
                wsat,
                n_points=2,
                color=color,
                linewidth=linewidth,
                **kwargs)
            return self.add_line(line)

        return [_draw_const_tdb_line(TDB) for TDB in tdb]

    def draw_const_density_lines(self,
                                 *rho,
                                 color='black',
                                 linewidth=.5,
                                 **kwargs):

        def foo(w, rho):
            V = (1 + w) / rho
            t = GetTDryBulbFromMoistAirVolumeAndHumRatio(V, w, self.pressure)
            if GetSatHumRatio(t, self.pressure) < w:
                return np.nan
            return GetMoistAirEnthalpy(t, w)

        def _draw_const_density_line(rho):
            line = ConstValueLine(rho, foo, color=color, linewidth=linewidth, **kwargs)
            return self.add_line(line)

        return [_draw_const_density_line(RHO) for RHO in rho]

    def draw_saturation_line(self, color='black', linewidth=2, **kwargs):

        def bound_fun():
            hmin, hmax = self.get_ybound()
            tmin = hmin / _CP_DRY_AIR
            tmax = hmax / _CP_DRY_AIR
            return tmin, tmax

        def calc_fun(t, _):
            w = GetSatHumRatio(t, self.pressure)
            h = GetSatAirEnthalpy(t, self.pressure)
            return w, h

        line = ParametricConstValueLine(None, calc_fun, bound_fun, color=color, linewidth=linewidth, **kwargs)
        return self.add_line(line)


class MollierProjection:

    def __init__(self, pressure: float = GetStandardAtmPressure(0.)) -> None:
        self._pressure = pressure

    def _as_mpl_axes(self):
        return MollierAxes, dict(pressure=self._pressure)


register_projection(MollierAxes)
