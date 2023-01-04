from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pytest

import mpl_mollier_axes as mol

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@pytest.fixture
def test_artifact_dir() -> Path:
    artifact_dir = Path('.') / 'testartifacts'
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


@pytest.fixture
def picture_dir(test_artifact_dir: Path) -> Path:
    picture_dir = test_artifact_dir / 'pictures'
    picture_dir.mkdir(parents=True, exist_ok=True)
    return picture_dir


@pytest.fixture
def fig(picture_dir: Path, request) -> Figure:
    fig = plt.figure()
    yield fig
    marker = request.node.get_closest_marker("fig_filename")
    if marker is not None:
        fname = picture_dir / marker.args[0]
        fig.savefig(fname)
    plt.close(fig)


@pytest.fixture
def mollier_axes(fig: Figure) -> mol.MollierAxes:
    ax = fig.add_subplot(projection='mollier')
    ax.set_xlim(0, 4e-2)
    ax.set_ylim(-1e4, 5e4)
    ax.grid(visible=True, which='both')
    yield ax
