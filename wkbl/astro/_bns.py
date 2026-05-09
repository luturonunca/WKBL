from . import component as comp
import numpy as np


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

        # All BNS times use RAMSES conformal code time (same convention as
        # particle_birth_time / t_sn2 / t_merge on disk).
        # Conversion to physical Gyr mirrors feedback.f90:
        #   delay_phys = delay_code * p.unitt / p.aexp**2 / SEC_PER_GYR
        # p.time is the conformal code time of the snapshot (from info file).
        _SEC_PER_GYR = 1e9 * 365.25 * 24.0 * 3600.0
        _code_to_gyr = p.unitt / p.aexp**2 / _SEC_PER_GYR
        try:
            _fam   = comp._yt_get_field(self._ad, [("io", "particle_family")]).value
            _bmask = (_fam == 6)
            _birth = comp._yt_get_field(
                self._ad, [("io", "particle_birth_time")]
            ).value[_bmask]
            self.age = (p.time - _birth) * _code_to_gyr
        except Exception:
            _birth = None
            self.age = np.zeros(_n)

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
            _t_sn2 = comp._yt_get_field(
                self._ad, [("io", "particle_t_sn2")]
            ).value[_bmask]
            self.delay_sn2 = (_t_sn2 - _birth) * _code_to_gyr
        except Exception:
            self.delay_sn2 = np.zeros(_n)

        try:
            _f = comp._yt_get_field(self._ad, [("bns", "particle_vkick2")])
            self.vkick2 = _f.to("km/s").value
        except Exception:
            try:
                self.vkick2 = np.array(_f.value) * p.simutokms
            except Exception:
                self.vkick2 = np.zeros(_n)

        try:
            _t_merge = comp._yt_get_field(
                self._ad, [("io", "particle_t_merge")]
            ).value[_bmask]
            self.delay_merge = (_t_merge - _birth) * _code_to_gyr
        except Exception:
            self.delay_merge = np.zeros(_n)

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
        self.vkick1      = self.vkick1[in_halo]
        self.delay_sn2   = self.delay_sn2[in_halo]
        self.vkick2      = self.vkick2[in_halo]
        self.delay_merge = self.delay_merge[in_halo]
        self.parent_id = self.parent_id[in_halo]
        self.m1        = self.m1[in_halo]
        self.r         = self.r[in_halo]
