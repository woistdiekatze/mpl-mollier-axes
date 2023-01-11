import numpy as np
from psychrolib import (GetHumRatioFromRelHum, GetMoistAirDensity, GetMoistAirEnthalpy, GetSatAirEnthalpy,
                        GetSatHumRatio, GetSatVapPres, GetTDryBulbFromMoistAirVolumeAndHumRatio)
from scipy.optimize import fsolve

from .base import BoundedConstValueLine, ParametricConstValueLine


class Isotherm(BoundedConstValueLine):

    def __init__(self,
                 tdb: float,
                 pressure: float,
                 *,
                 xmin: float = None,
                 xmax: float = None,
                 n_points: int = 2,
                 **kwargs) -> None:

        def calc_fun(w, t):
            return GetMoistAirEnthalpy(t, w)

        xmin = xmin or 0
        xmax = xmax or GetSatHumRatio(tdb, pressure)
        super().__init__(tdb, calc_fun, xmin, xmax, n_points=n_points, **kwargs)


_CP_DRY_AIR = 1.006e3  # J / kg K


class ConstRhLine(ParametricConstValueLine):

    def __init__(self, rh: float, pressure: float, *, n_points: int = 100, **kwargs) -> None:

        def bound_fun():
            tmin, tmax = (h / _CP_DRY_AIR for h in self.axes.get_ybound())
            return tmin, tmax

        def calc_fun(t, rh):
            w = GetHumRatioFromRelHum(t, rh, pressure)
            h = GetMoistAirEnthalpy(t, w)
            return w, h

        super().__init__(rh, calc_fun, bound_fun, n_points=n_points, **kwargs)


class SaturationLine(ParametricConstValueLine):

    def __init__(self, pressure: float, *, n_points: int = 100, **kwargs) -> None:

        def bound_fun():
            tmin, tmax = (h / _CP_DRY_AIR for h in self.axes.get_ybound())
            return tmin, tmax

        def calc_fun(t, _):
            w = GetSatHumRatio(t, pressure)
            h = GetSatAirEnthalpy(t, pressure)
            return w, h

        zorder = kwargs.pop('zorder', 2.01)
        super().__init__(None, calc_fun, bound_fun, n_points=n_points, zorder=zorder, **kwargs)


def calc_saturation_data(pressure, num: int = 100):

    def _obj(t):
        return (GetSatVapPres(t) - pressure)**2

    def get_max_t():
        max_t, = fsolve(_obj, 100)
        return max_t

    t = np.linspace(-100, get_max_t(), num=num, endpoint=False)
    w = np.vectorize(GetSatHumRatio)(t, pressure)

    return w, t


class ConstDensityLine(BoundedConstValueLine):

    def __init__(self, rho: float, pressure: float, *, xmin: float = None, xmax: float = None, **kwargs) -> None:

        wp, tp = calc_saturation_data(pressure)
        rhop = np.vectorize(GetMoistAirDensity)(tp, wp, pressure)

        def get_w(rho):
            return np.interp(rho, np.flip(rhop), np.flip(wp))

        def calc_fun(w, rho):
            V = (1 + w) / rho
            t = GetTDryBulbFromMoistAirVolumeAndHumRatio(V, w, pressure)
            return GetMoistAirEnthalpy(t, w)

        xmin = xmin or 0
        xmax = xmax or get_w(rho)

        super().__init__(rho, calc_fun, xmin, xmax, **kwargs)
