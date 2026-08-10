import numpy as np
from scipy.optimize import fsolve

from ..backends import PsychrometricBackend, get_default_backend
from .base import BoundedConstValueLine, ParametricConstValueLine

_CP_DRY_AIR = 1.006e3  # J / kg K

# ConstRhLine and SaturationLine parametrize over dry-bulb temperature `t` and
# need a `t` domain wide enough to cover the visible axes. Note that on this
# skewed-y projection, `ax.get_ybound()` bounds the *skewed* enthalpy
# h' = h - L*w (the skew is applied before the axes' view limits are used to
# normalize data, see SkewYAxes._adapt_transforms), not the raw enthalpy h.
# Since h' = t*(cp_dry_air + w*cp_water_vapour) and w >= 0, `h' / cp_dry_air`
# is always at least as far from zero as the true t root - i.e. this estimate
# is a safe, if slightly loose, bound: matplotlib clips the small excess.


class Isotherm(BoundedConstValueLine):

    def __init__(self,
                 tdb: float,
                 pressure: float,
                 *,
                 xmin: float = None,
                 xmax: float = None,
                 n_points: int = 2,
                 backend: PsychrometricBackend = None,
                 **kwargs) -> None:
        backend = backend if backend is not None else get_default_backend()

        def calc_fun(w, t):
            return backend.get_moist_air_enthalpy(t, w)

        xmin = xmin or 0
        xmax = xmax or backend.get_sat_hum_ratio(tdb, pressure)
        super().__init__(tdb, calc_fun, xmin, xmax, n_points=n_points, **kwargs)


class ConstRhLine(ParametricConstValueLine):

    def __init__(self, rh: float, *, n_points: int = 100, backend: PsychrometricBackend = None, **kwargs) -> None:
        backend = backend if backend is not None else get_default_backend()

        def calc_fun(t, rh):
            w = backend.get_hum_ratio_from_rel_hum(t, rh, self.axes.pressure)
            h = backend.get_moist_air_enthalpy(t, w)
            return w, h

        def bound_fun():
            tmin, tmax = (h / _CP_DRY_AIR for h in self.axes.get_ybound())
            return tmin, tmax

        super().__init__(
            rh, calc_fun, bound_fun, n_points=n_points, extra_key_fun=lambda: (self.axes.pressure, ), **kwargs
        )


class SaturationLine(ParametricConstValueLine):

    def __init__(self, *, n_points: int = 100, backend: PsychrometricBackend = None, **kwargs) -> None:
        backend = backend if backend is not None else get_default_backend()

        def calc_fun(t, _):
            w = backend.get_sat_hum_ratio(t, self.axes.pressure)
            h = backend.get_sat_air_enthalpy(t, self.axes.pressure)
            return w, h

        def bound_fun():
            tmin, tmax = (h / _CP_DRY_AIR for h in self.axes.get_ybound())
            return tmin, tmax

        zorder = kwargs.pop('zorder', 2.01)
        super().__init__(
            None, calc_fun, bound_fun, n_points=n_points, zorder=zorder,
            extra_key_fun=lambda: (self.axes.pressure, ), **kwargs
        )


def calc_saturation_data(pressure, num: int = 100, *, backend: PsychrometricBackend = None):
    backend = backend if backend is not None else get_default_backend()

    def _obj(t):
        return (backend.get_sat_vap_pres(float(t[0])) - pressure)**2

    def get_max_t():
        max_t, = fsolve(_obj, 100)
        return max_t

    t = np.linspace(-100, get_max_t(), num=num, endpoint=False)
    w = np.vectorize(backend.get_sat_hum_ratio)(t, pressure)

    return w, t


class ConstDensityLine(BoundedConstValueLine):

    def __init__(self,
                 rho: float,
                 pressure: float,
                 *,
                 xmin: float = None,
                 xmax: float = None,
                 backend: PsychrometricBackend = None,
                 **kwargs) -> None:
        backend = backend if backend is not None else get_default_backend()

        wp, tp = calc_saturation_data(pressure, backend=backend)
        rhop = np.vectorize(backend.get_moist_air_density)(tp, wp, pressure)

        def get_w(rho):
            return np.interp(rho, np.flip(rhop), np.flip(wp))

        def calc_fun(w, rho):
            V = (1 + w) / rho
            t = backend.get_tdb_from_moist_air_volume_and_hum_ratio(V, w, pressure)
            return backend.get_moist_air_enthalpy(t, w)

        xmin = xmin or 0
        xmax = xmax or get_w(rho)

        super().__init__(rho, calc_fun, xmin, xmax, **kwargs)
