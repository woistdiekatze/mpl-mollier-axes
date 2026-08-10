import numpy as np
from matplotlib.lines import Line2D


class ParametricConstValueLine(Line2D):

    def __init__(self, const_val, calc_fun, bound_fun, *, n_points: int = 100, extra_key_fun=None, **kwargs) -> None:
        super().__init__([0, 1], [0, 1], **kwargs)
        self._const_val = const_val
        self._calc_fun = np.vectorize(calc_fun)
        self._bound_fun = bound_fun
        self._n_points = n_points
        self._extra_key_fun = extra_key_fun if extra_key_fun is not None else (lambda: ())
        self._cache_key = None

    def get_xbound(self):
        return self._bound_fun()

    def _get_cache_key(self):
        # Bundles everything `recalc` depends on so `draw` can skip the
        # (potentially expensive) recomputation when nothing has changed
        # since the last draw, e.g. repeated redraws with an unchanged view.
        return (self.get_xbound(), self._extra_key_fun())

    def recalc(self):
        tmin, tmax = self.get_xbound()
        t = np.linspace(tmin, tmax, self._n_points)
        self.set_data(*self._calc_fun(t, self._const_val))

    def draw(self, renderer):
        key = self._get_cache_key()
        if key != self._cache_key:
            self._cache_key = key
            self.recalc()
        super().draw(renderer)


class ConstValueLine(ParametricConstValueLine):
    """
    A helper class that implements `~.Axes.axline`, by recomputing the artist
    transform at draw time.
    """

    def __init__(self, const_val, calc_fun, n_points: int = 100, **kwargs) -> None:
        super().__init__(const_val, calc_fun, lambda: self.axes.get_xbound(), n_points=n_points, **kwargs)

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
        return (
            max(self._xmin, xmin) if self._xmin is not None else xmin,
            min(self._xmax, xmax) if self._xmax is not None else xmax
        )
