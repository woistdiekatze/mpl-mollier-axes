"""Pluggable interface for the psychrometric property calculations used to draw
Mollier chart lines, decoupling `mpl_mollier_axes.lines.psychrometric` from any
single calculation library (e.g. to swap PsychroLib for CoolProp)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class PsychrometricBackend(Protocol):

    def get_hum_ratio_from_rel_hum(self, tdb: float, rh: float, pressure: float) -> float:
        ...  # pragma: no cover

    def get_moist_air_enthalpy(self, tdb: float, hum_ratio: float) -> float:
        ...  # pragma: no cover

    def get_sat_air_enthalpy(self, tdb: float, pressure: float) -> float:
        ...  # pragma: no cover

    def get_sat_hum_ratio(self, tdb: float, pressure: float) -> float:
        ...  # pragma: no cover

    def get_sat_vap_pres(self, tdb: float) -> float:
        ...  # pragma: no cover

    def get_moist_air_density(self, tdb: float, hum_ratio: float, pressure: float) -> float:
        ...  # pragma: no cover

    def get_tdb_from_moist_air_volume_and_hum_ratio(self, volume: float, hum_ratio: float, pressure: float) -> float:
        ...  # pragma: no cover


class PsychrolibBackend:
    """Default backend, delegating to the PsychroLib library (ASHRAE correlations)."""

    def __init__(self) -> None:
        import psychrolib
        psychrolib.SetUnitSystem(psychrolib.SI)
        self._lib = psychrolib

    def get_hum_ratio_from_rel_hum(self, tdb, rh, pressure):
        return self._lib.GetHumRatioFromRelHum(tdb, rh, pressure)

    def get_moist_air_enthalpy(self, tdb, hum_ratio):
        return self._lib.GetMoistAirEnthalpy(tdb, hum_ratio)

    def get_sat_air_enthalpy(self, tdb, pressure):
        return self._lib.GetSatAirEnthalpy(tdb, pressure)

    def get_sat_hum_ratio(self, tdb, pressure):
        return self._lib.GetSatHumRatio(tdb, pressure)

    def get_sat_vap_pres(self, tdb):
        return self._lib.GetSatVapPres(tdb)

    def get_moist_air_density(self, tdb, hum_ratio, pressure):
        return self._lib.GetMoistAirDensity(tdb, hum_ratio, pressure)

    def get_tdb_from_moist_air_volume_and_hum_ratio(self, volume, hum_ratio, pressure):
        return self._lib.GetTDryBulbFromMoistAirVolumeAndHumRatio(volume, hum_ratio, pressure)


_DEFAULT_BACKEND = PsychrolibBackend()


def get_default_backend() -> PsychrometricBackend:
    return _DEFAULT_BACKEND
