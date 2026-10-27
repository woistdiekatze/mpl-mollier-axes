# Design outline: hastate-backed lines and display-space sampling

Status: **draft, decisions recorded** (tracking issue: #11). Companion to `woistdiekatze/hastate` `docs/design/lines.md`. No code changes accompany this document.

## Decisions

- **Hard dependency on `hastate`.** All psychrometric calculation goes through it; no local backend protocol and no duck-typing fallback for lines.
- Quadrature is numpy-only in `hastate` (scipy accelerator: woistdiekatze/hastate#2).
- Backends expose capability flags; unsupported properties raise `UnsupportedPropertyError` (relevant for tick/label code that might request optional properties).
- Altitude / standard-pressure helper lives in `hastate` (woistdiekatze/hastate#3); this library drops its direct `psychrolib` use in favour of it.
- Packaging (PyPI vs. PEP 621 vs. both) still open, see section 7.

## 1. Current state (commit d9b582f)

- `backends.py` defines `PsychrometricBackend` (psychrolib-shaped, scalar calls) and a `PsychrolibBackend` default.
- `lines/base.py`: `ParametricConstValueLine` samples `np.linspace(tmin, tmax, n_points)` and recomputes only when axes bounds / extra key change.
- `lines/psychrometric.py`: `Isotherm` (2 points, exact because isotherms are straight in (W, H)), `ConstRhLine`, `SaturationLine` (100 points over `t`, bounds derived from `ax.get_ybound() / cp_dry_air`), `ConstDensityLine` (fsolve + interpolation table).
- `axes/MollierAxes.py` imports `psychrolib` directly (`SetUnitSystem`, `GetStandardAtmPressure`) in addition to the backend abstraction, so PsychroLib is a de-facto hard dependency regardless of `backend=`.
- `pyproject.toml`: `PsychroLib` and `scipy` are mandatory; CoolProp is an optional extra.

## 2. Target responsibilities

| Concern | Owner |
|---|---|
| Psychrometric property math, backends, root-finding, integration | `hastate` |
| Constant-property curves as parametric `Line` objects | `hastate` |
| Axes, skew transform, ticks, matplotlib artists | this library |
| **Display-space sampling and clipping** | this library |

## 3. Display-space sampling

Replace fixed `n_points` linspace with refinement driven by the axes:

1. If `line.is_straight`: use the two domain endpoints (current isotherm behaviour, generalized to const-H and const-W).
2. Otherwise: start from a coarse set of parameter values, evaluate `line.at(s)` (vectorized), transform to display coordinates via `ax.transData`, and subdivide segments whose midpoint deviates from the chord by more than a pixel tolerance (e.g. 0.25 px), up to a max depth.
3. Clip the parameter range to the visible region before refining, so zooming in does not waste points off-screen. This replaces the `get_ybound() / _CP_DRY_AIR` estimate with a proper intersection of the line with the view box.
4. Cache on (view bounds, pressure, figure dpi/size) instead of bounds and pressure only, since pixel tolerance depends on the figure.

Sampling for display is independent of any numerical integration done in `hastate`.

## 4. Backend selection

- Resolution is delegated to `hastate` (explicit `backend=` > `hastate.set_backend()` > `HASTATE_BACKEND` > `MissingBackendError`).
- `MollierAxes.__init__` resolves the backend once and stores it on the instance; lines receive it from the axes. No module-level global in this library.
- Remove direct `psychrolib` usage from `MollierAxes`; take the standard-pressure helper from `hastate` (#3 there).
- `PsychrometricBackend` and `PsychrolibBackend` are removed from this library (re-exported with a deprecation warning for one release if needed).
- Packaging: `mpl-mollier-axes[coolprop]` / `[psychrolib]` forward to `hastate[...]`; see section 7.

## 5. Plotting states and processes

Accept anything exposing `W` and `H` (`Protocol`), so `HaState`, plain dataclasses and tuples work. This applies to plotted **points**; lines come from `hastate` and are not duck-typed.

```python
ax.plot_state(point, **style)
ax.plot_process(a, b, **style)     # straight segment, or a Line segment if provided
```

## 6. Migration / compatibility

- Breaking for 0.2.0: PsychroLib and scipy no longer mandatory here (they follow from the chosen `hastate` extra); `PsychrometricBackend` moves to `hastate`. CHANGELOG entry required.
- `n_points` stays accepted for one release, with a deprecation warning, as a cap on refinement.
- Tests: parametrize over installed backends; add pixel-tolerance tests with fixed figure size.

## 7. Packaging and remaining questions

Extras forwarding to `hastate[...]` depends on the packaging decision recorded in woistdiekatze/hastate#1.

Remaining:

1. Does `scipy` remain needed here once `fsolve` usage moves to `hastate`? Expected: no.
2. Redraw cost with CoolProp: needs a benchmark once vectorized backends exist.
3. README.rst is empty and the Python 3.8 classifier contradicts `python = "^3.10"`; fix alongside.
