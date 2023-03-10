from __future__ import annotations

from contextlib import ExitStack
from math import degrees, radians
from typing import TYPE_CHECKING

import matplotlib.axis as maxis
import matplotlib.spines as mspines
import matplotlib.transforms as transforms
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from typing import Tuple


class SkewYProjection:

    def __init__(self, skewy_deg: float) -> None:
        self.skewy = radians(skewy_deg)

    def _as_mpl_axes(self) -> Tuple[type, dict]:
        return SkewYAxes, dict(skewy=self.skewy)


# The sole purpose of this class is to look at the upper, lower, or total
# interval as appropriate and see what parts of the tick to draw, if any.
class SkewYTick(maxis.YTick):

    def draw(self, renderer):
        # When adding the callbacks with `stack.callback`, we fetch the current
        # visibility state of the artist with `get_visible`; the ExitStack will
        # restore these states (`set_visible`) at the end of the block (after
        # the draw).
        with ExitStack() as stack:
            for artist in [self.gridline, self.tick1line, self.tick2line, self.label1, self.label2]:
                stack.callback(artist.set_visible, artist.get_visible())

            needs_left = transforms.interval_contains(self.axes.left_ylim, self.get_loc())
            self.tick1line.set_visible(self.tick1line.get_visible() and needs_left)
            self.label1.set_visible(self.label1.get_visible() and needs_left)

            needs_right = transforms.interval_contains(self.axes.right_ylim, self.get_loc())
            self.tick2line.set_visible(self.tick2line.get_visible() and needs_right)
            self.label2.set_visible(self.label2.get_visible() and needs_right)

            needs_grid = transforms.interval_contains(self.get_view_interval(), self.get_loc())
            self.gridline.set_visible(self.gridline.get_visible() and needs_grid)

            super().draw(renderer)

    def update_position(self, loc: int) -> None:
        super().update_position(loc)

        ax: SkewYAxes = self.axes

        angle, = ax.transData.transform_angles((0, ), ((0, 0), ), radians=True)
        angle_deg = degrees(angle)
        trans = transforms.Affine2D().rotate(angle)

        for tickline in (self.tick1line, self.tick2line):
            tickline._marker._user_transform = trans

        for ticklabel in (self.label1, self.label2):
            ticklabel.set_rotation_mode('anchor')
            ticklabel.set_rotation(angle_deg)

    def get_view_interval(self):
        return self.axes.yaxis.get_view_interval()


# This class exists to provide two separate sets of intervals to the tick,
# as well as create instances of the custom tick
class SkewYAxis(maxis.YAxis):

    _tick_class = SkewYTick

    def get_view_interval(self):
        ax: SkewYAxes = self.axes
        ll, lu = ax.left_ylim
        rl, ru = ax.right_ylim

        lower = min(ll, rl)
        upper = max(lu, ru)
        return lower, upper


# This class exists to calculate the separate data range of the
# right Y-axis and draw the spine there. It also provides this range
# to the Y-axis artist for ticking and gridlines
class SkewSpine(mspines.Spine):

    def _adjust_location(self):
        pts = self._path.vertices
        ax: SkewYAxes = self.axes
        if self.spine_type == 'right':
            pts[:, 1] = ax.right_ylim
        else:
            pts[:, 1] = ax.left_ylim


# This class handles registration of the skew-yaxes as a projection as well
# as setting up the appropriate transformations. It also overrides standard
# spines and axes instances as appropriate.
class SkewYAxes(Axes):

    def __init__(self, *args, skewy: float, **kwargs) -> None:
        self.set_angle(skewy)
        super().__init__(*args, **kwargs)

    def set_angle(self, skewy: float) -> None:
        self._skewy = skewy
        self.transAffine = transforms.Affine2D().skew(0, skewy)

    def _init_axis(self):
        # Taken from Axes and modified to use our modified Y-axis
        self.xaxis = maxis.XAxis(self)
        self.spines.top.register_axis(self.xaxis)
        self.spines.bottom.register_axis(self.xaxis)
        self.yaxis = SkewYAxis(self)
        self.spines.left.register_axis(self.yaxis)
        self.spines.right.register_axis(self.yaxis)

    def _gen_axes_spines(self):
        spines = {
            'top': mspines.Spine.linear_spine(self, 'top'),
            'bottom': mspines.Spine.linear_spine(self, 'bottom'),
            'left': SkewSpine.linear_spine(self, 'left'),
            'right': SkewSpine.linear_spine(self, 'right')
        }
        return spines

    def _set_lim_and_transforms(self):
        """
        This is called once when the plot is created to set up all the
        transforms for the data, text and grids.
        """
        # Get the standard transform setup from the Axes base class
        super()._set_lim_and_transforms()
        self._adapt_transforms()

    def _adapt_transforms(self):
        # Create the full transform from Data to Pixels
        self.transDataToAxes = self.transScale + self.transAffine + self.transLimits
        self.transData = self.transDataToAxes + self.transAxes

        # Blended transforms like this need to have the skewing applied using
        # both axes, in axes coords like before.

        self._xaxis_transform = (
            transforms.blended_transform_factory(
                self.transScale + self.transLimits,
                transforms.IdentityTransform()
            ) + self.transAxes)

        self._yaxis_transform = (
            transforms.blended_transform_factory(
                transforms.BboxTransformTo(self.viewLim),
                transforms.IdentityTransform()
            ) + self.transData)

    @property
    def left_ylim(self):
        return self.axes.viewLim.intervaly

    @property
    def right_ylim(self):
        pts = [[1., 0.], [1., 1.]]
        return self.transDataToAxes.inverted().transform(pts)[:, 1]
