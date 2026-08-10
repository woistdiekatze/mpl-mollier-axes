import subprocess
import sys

import matplotlib.pyplot as plt

from mpl_mollier_axes import MollierProjection
from mpl_mollier_axes.backends import PsychrolibBackend, get_default_backend


class RecordingBackend(PsychrolibBackend):
    """Wraps the default backend but records every call, to verify that a
    custom backend passed in by a user is actually the one being used."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def get_hum_ratio_from_rel_hum(self, tdb, rh, pressure):
        self.calls.append('get_hum_ratio_from_rel_hum')
        return super().get_hum_ratio_from_rel_hum(tdb, rh, pressure)

    def get_sat_hum_ratio(self, tdb, pressure):
        self.calls.append('get_sat_hum_ratio')
        return super().get_sat_hum_ratio(tdb, pressure)


def test_get_default_backend_is_a_singleton():
    assert get_default_backend() is get_default_backend()


def test_default_backend_is_lazily_initialized():
    # Run in a subprocess so `mpl_mollier_axes.backends` is imported fresh:
    # merely importing it must not construct a PsychrolibBackend instance.
    script = (
        "import mpl_mollier_axes.backends as b\n"
        "assert b._default_backend is None, 'backend constructed on import'\n"
        "backend = b.get_default_backend()\n"
        "assert b._default_backend is backend\n"
        "assert isinstance(backend, b.PsychrolibBackend)\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_custom_backend_is_used_for_const_rh_lines():
    backend = RecordingBackend()
    fig = plt.figure()
    ax = fig.add_subplot(projection=MollierProjection(backend=backend))
    ax.set_xlim(0, 4e-2)
    ax.set_ylim(-1e4, 5e4)

    ax.draw_const_rh_lines(0.5)
    fig.canvas.draw()

    assert 'get_hum_ratio_from_rel_hum' in backend.calls


def test_custom_backend_is_used_for_saturation_line():
    backend = RecordingBackend()
    fig = plt.figure()
    ax = fig.add_subplot(projection=MollierProjection(backend=backend))
    ax.set_xlim(0, 4e-2)
    ax.set_ylim(-1e4, 5e4)

    ax.draw_saturation_line()
    fig.canvas.draw()

    assert 'get_sat_hum_ratio' in backend.calls
