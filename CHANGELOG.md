# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added methods to draw lines of constant propertiy values
- Added `mpl_mollier_axes.backends.PsychrometricBackend`, a pluggable interface for the
  psychrometric property calculations used to draw chart lines. `MollierAxes` and
  `MollierProjection` accept a `backend=` argument; defaults to PsychroLib.

### Changed

- YTicks and tick labels are rotated by the skew angle of the axes.
- `draw_const_rh_lines`/`draw_const_tdb_lines`/`draw_const_density_lines`/`draw_const_h_lines`/
  `draw_const_w_lines` now default to `zorder=1` and `draw_saturation_line` to `zorder=1.5`, both
  below the default z-order of regular plotted data (2), so lines of constant property value sit
  behind data plotted onto the chart afterwards, similar to axis gridlines.
- `ConstRhLine` and `SaturationLine` (and `draw_const_rh_lines`/`draw_saturation_line`) now read
  the axes' `pressure` live at each redraw instead of capturing it at construction time, so
  already-drawn lines pick up a later `ax.pressure = ...` change automatically. As part of this,
  `ConstRhLine` and `SaturationLine` no longer take `pressure` as a constructor argument.

### Fixed

- Lines of constant property value no longer unconditionally recompute their data on every draw;
  they now cache on the axes bounds (and, for `ConstRhLine`/`SaturationLine`, pressure) they
  depend on and only recompute when those actually change.

## [0.1.0] - 2023-01-01

### Added

- Created subclasses of matplotlib Axes to facilitate creation of mollier plots

[unreleased]: https://github.com/woistdiekatze/mpl-mollier-axes/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/woistdiekatze/mpl-mollier-axes/releases/tag/v0.1.0
