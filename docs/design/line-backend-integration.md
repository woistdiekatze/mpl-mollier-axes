# Design outline: hastate-backed lines and display-space sampling

Status: **draft for review** (tracking issue: #11). Companion to `woistdiekatze/hastate` `docs/design/lines.md`. No code changes accompany this document.

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
- Remove direct `psychrolib` usage from `MollierAxes`; take the standard-pressure helper from `hastate`.
- Packaging: `mpl-mollier-axes[coolprop]` / `[psychrolib]` forward to `hastate[...]`. This needs PEP 621 `[project.optional-dependencies]` or PyPI releases (Poetry git-dependency extras forwarding is awkward).

## 5. Plotting states and processes

Accept anything exposing `W` and `H` (`Protocol`), so `HaState`, plain dataclasses and tuples work without importing `hastate` types:

```python
ax.plot_state(point, **style)
ax.plot_process(a, b, **style)     # straight segment, or a Line segment if provided
```

## 6. Migration / compatibility

- Breaking for 0.2.0: PsychroLib no longer a mandatory dependency; `PsychrometricBackend` moves to (or is re-exported from) `hastate`. CHANGELOG entry required.
- `n_points` stays accepted for one release, with a deprecation warning, as a cap on refinement.
- Tests: parametrize over installed backends; add pixel-tolerance tests with fixed figure size.

## 7. Open questions

1. Hard dependency on `hastate` vs. a thin local protocol plus duck typing. This document assumes the hard dependency (single calculation source).
2. Does `scipy` remain needed here once `fsolve` usage moves to `hastate`?
3. Redraw cost with CoolProp: needs a benchmark once vectorized backends exist.
4. README.rst is empty and the Python 3.8 classifier contradicts `python = "^3.10"`; fix alongside.
