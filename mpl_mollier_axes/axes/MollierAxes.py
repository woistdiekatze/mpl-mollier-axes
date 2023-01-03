from math import atan

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.projections import register_projection
from psychrolib import (SI, GetHumRatioFromRelHum, GetMoistAirEnthalpy,
                        GetSatAirEnthalpy, GetSatHumRatio,
                        GetStandardAtmPressure,
                        GetTDryBulbFromMoistAirVolumeAndHumRatio,
                        SetUnitSystem)

from .SkewYAxes import SkewYAxes

SetUnitSystem(SI)
_H_EVAP_H2O_0CELSIUS = 2.501e6  # J / kg
_CP_DRY_AIR = 1.006e3  # J / kg K


class ParametricConstValueLine(Line2D):

    def __init__(self,
                 const_val,
                 calc_fun,
                 bound_fun,
                 *,
                 n_points: int = 100,
                 **kwargs) -> None:
        super().__init__([0, 1], [0, 1], **kwargs)
        self._const_val = const_val
        self._calc_fun = np.vectorize(calc_fun)
        self._bound_fun = bound_fun
        self._n_points = n_points

    def recalc(self):
        tmin, tmax = self._bound_fun()
        t = np.linspace(tmin, tmax, self._n_points)
        self.set_data(*self._calc_fun(t, self._const_val))

    def draw(self, renderer):
        # return super().draw(renderer)
        self._transformed_path = None  # Force regen.

        self.recalc()

        super().draw(renderer)


class ConstValueLine(ParametricConstValueLine):
    """
    A helper class that implements `~.Axes.axline`, by recomputing the artist
    transform at draw time.
    """

    def __init__(self,
                 const_val,
                 calc_fun,
                 n_points: int = 100,
                 **kwargs) -> None:
        super().__init__(const_val,
                         calc_fun,
                         lambda: self.axes.get_xbound(),
                         n_points=n_points,
                         **kwargs)

    def get_xbound(self):
        return self._bound_fun()

    def recalc(self):
        tmin, tmax = self.get_xbound()
        t = np.linspace(tmin, tmax, self._n_points)
        self.set_data(t, self._calc_fun(t, self._const_val))


class BoundedConstValueLine(ConstValueLine):

    def __init__(self, const_val, calc_fun, xmin, xmax, **kwargs) -> None:
        super().__init__(const_val, calc_fun, **kwargs)
        self._xmin = xmin
        self._xmax = xmax

    def get_xbound(self):
        xmin, xmax = super().get_xbound()
        return max(self._xmin, xmin) if self._xmin is not None else xmin, min(
            self._xmax, xmax) if self._xmax is not None else xmax


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
        self._sat = self.draw_saturation_line()

    @property
    def pressure(self):
        return self._pressure

    @pressure.setter
    def pressure(self, pressure: float):
        if pressure <= 0:
            raise ValueError("pressure must be greater than zero.")
        self._pressure = pressure

    def draw_const_h_lines(self, *h, **kwargs):
        return [self.axhline(H, **kwargs) for H in h]

    def draw_const_w_lines(self, *w, **kwargs):
        return [self.axvline(W, **kwargs) for W in w]

    def draw_const_rh_lines(self, *rh, **kwargs):

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

            line = ParametricConstValueLine(rh, calc_fun, bound_fun, **kwargs)
            return self.add_line(line)

        return [_draw_const_rh_line(RH) for RH in rh]

    def draw_const_tdb_lines(self, *tdb, **kwargs):

        def _draw_const_tdb_line(tdb):
            wsat = GetSatHumRatio(tdb, self.pressure)
            line = BoundedConstValueLine(tdb,
                                      lambda w, t: GetMoistAirEnthalpy(t, w),
                                      0,
                                      wsat,
                                      n_points=2,
                                      **kwargs)
            return self.add_line(line)

        return [_draw_const_tdb_line(TDB) for TDB in tdb]

    def draw_const_density_lines(self, *rho, **kwargs):

        def foo(w, rho):
            V = (1 + w) / rho
            t = GetTDryBulbFromMoistAirVolumeAndHumRatio(V, w, self.pressure)
            if GetSatHumRatio(t, self.pressure) < w:
                return np.nan
            return GetMoistAirEnthalpy(t, w)

        def _draw_const_density_line(rho):
            line = ConstValueLine(rho, foo, **kwargs)
            return self.add_line(line)

        return [_draw_const_density_line(RHO) for RHO in rho]

    def draw_saturation_line(self, **kwargs):

        def bound_fun():
            hmin, hmax = self.get_ybound()
            tmin = hmin / _CP_DRY_AIR
            tmax = hmax / _CP_DRY_AIR
            return tmin, tmax

        def calc_fun(t, _):
            w = GetSatHumRatio(t, self.pressure)
            h = GetSatAirEnthalpy(t, self.pressure)
            return w, h

        line = ParametricConstValueLine(None, calc_fun, bound_fun, **kwargs)
        return self.add_line(line)


class MollierProjection:

    def __init__(self, pressure: float = GetStandardAtmPressure(0.)) -> None:
        self._pressure = pressure

    def _as_mpl_axes(self):
        return MollierAxes, dict(pressure=self._pressure)


register_projection(MollierAxes)
