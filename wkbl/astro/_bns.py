from . import component as comp
import numpy as np
from yt.frontends.ramses.io import convert_ramses_conformal_time_to_physical_age


class _bns(comp.Component):
    def __init__(self, file_path, p, **kwargs):
        comov = kwargs.get('comov', False)
        ds = kwargs.get("ds")
        ad = kwargs.get("ad")
        tcur_code = kwargs.get("tcur_code", None)
        super().__init__(file_path, "bns", p, comov=comov, ds=ds, ad=ad)
        _n = len(self.mass)

        try:
            self.stage = np.array(
                comp._yt_get_field(self._ad, [("bns", "particle_tag")]).value,
                dtype=np.int32,
            )
        except Exception:
            self.stage = np.zeros(_n, dtype=np.int32)

        _SEC_PER_GYR = 1e9 * 365.25 * 24.0 * 3600.0
        # Age since birth in Gyr — yt uses the full conformal-time lookup table.
        try:
            self.age = np.array(
                comp._yt_get_field(self._ad, [("bns", "star_age")]).to("Gyr").value
            )
        except Exception:
            self.age = np.zeros(_n)
        # Family mask selects BNS particles from the ("io",...) all-particle arrays.
        try:
            _fam   = comp._yt_get_field(self._ad, [("io", "particle_family")]).value
            _bmask = (_fam == 6)
        except Exception:
            _bmask = None
        # Snapshot physical time in Gyr — used to form absolute event times.
        _t_cur_gyr = float(self._ds.current_time.to("Gyr"))

        try:
            self.metal = np.array(
                comp._yt_get_field(
                    self._ad,
                    [("bns", "particle_metallicity"), ("bns", "particle_metal")],
                ).value
            )
        except Exception:
            self.metal = np.zeros(_n)

        try:
            self.Eu = np.array(
                comp._yt_get_field(
                    self._ad,
                    [("bns", "particle_bns_enrichment")],
                ).value
            )
        except Exception:
            self.Eu = np.zeros(_n)

        try:
            _f = comp._yt_get_field(self._ad, [("bns", "particle_vkick1")])
            self.vkick1 = _f.to("km/s").value
        except Exception:
            try:
                self.vkick1 = np.array(_f.value) * p.simutokms
            except Exception:
                self.vkick1 = np.zeros(_n)

        try:
            _tau_sn2 = comp._yt_get_field(
                self._ad, [("io", "particle_t_sn2")]
            ).value[_bmask]
            _age_sn2 = convert_ramses_conformal_time_to_physical_age(self._ds, _tau_sn2)
            self.t_sn2 = _t_cur_gyr - _age_sn2 * p.unitt / _SEC_PER_GYR
        except Exception:
            self.t_sn2 = np.zeros(_n)

        try:
            _f = comp._yt_get_field(self._ad, [("bns", "particle_vkick2")])
            self.vkick2 = _f.to("km/s").value
        except Exception:
            try:
                self.vkick2 = np.array(_f.value) * p.simutokms
            except Exception:
                self.vkick2 = np.zeros(_n)

        try:
            _tau_merge = comp._yt_get_field(
                self._ad, [("io", "particle_t_merge")]
            ).value[_bmask]
            _age_merge = convert_ramses_conformal_time_to_physical_age(self._ds, _tau_merge)
            self.t_merge = _t_cur_gyr - _age_merge * p.unitt / _SEC_PER_GYR
        except Exception:
            self.t_merge = np.zeros(_n)

        try:
            self.parent_id = np.array(
                comp._yt_get_field(self._ad, [("bns", "particle_parent_id")]).value,
                dtype=np.int64,
            )
        except Exception:
            self.parent_id = np.zeros(_n, dtype=np.int64)

        try:
            _f = comp._yt_get_field(self._ad, [("bns", "particle_m1")])
            self.m1 = _f.to("Msun").value
        except Exception:
            try:
                self.m1 = np.array(_f.value) * p.simutoMsun
            except Exception:
                self.m1 = np.zeros(_n)

    def halo_Only(self, center, n, r200, simple=False):
        super().halo_Only(center, n, r200, simple=simple)
        in_halo = np.where(self.r <= n * r200)
        self.stage   = self.stage[in_halo]
        self.age     = self.age[in_halo]
        self.metal   = self.metal[in_halo]
        self.Eu      = self.Eu[in_halo]
        self.vkick1  = self.vkick1[in_halo]
        self.t_sn2   = self.t_sn2[in_halo]
        self.vkick2  = self.vkick2[in_halo]
        self.t_merge = self.t_merge[in_halo]
        self.parent_id = self.parent_id[in_halo]
        self.m1        = self.m1[in_halo]
        self.r         = self.r[in_halo]
