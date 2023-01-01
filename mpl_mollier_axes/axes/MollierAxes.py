from math import atan

from matplotlib.projections import register_projection

from .SkewYAxes import SkewYAxes

_H_EVAP_H2O_0CELSIUS = 2.501e6  # J / kg


class MollierAxes(SkewYAxes):
    name = 'mollier'

    def __init__(self, *args, **kwargs) -> None:
        # The skew transformation will be carried out in data coordinates.
        # Thus the skew angle corresponds to the heat of evaporation of pure water at 0 dgrees celsius.
        super().__init__(*args, skewy=-atan(_H_EVAP_H2O_0CELSIUS), **kwargs)


register_projection(MollierAxes)
