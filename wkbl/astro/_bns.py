from . import component as comp
import numpy as np


class _bns(comp.Component):
    def __init__(self, file_path, p, **kwargs):
        comov = kwargs.get('comov', False)
        ds = kwargs.get("ds")
        ad = kwargs.get("ad")
        super().__init__(file_path, "bns", p, comov=comov, ds=ds, ad=ad)
        _n = len(self.mass)

        try:
            self.stage = np.array(
                comp._yt_get_field(self._ad, [("bns", "particle_tag")]).value,
                dtype=np.int32,
            )
        except KeyError:
            self.stage = np.zeros(_n, dtype=np.int32)

        try:
            age = comp._yt_get_field(
                self._ad,
                [("bns", "particle_age"), ("bns", "particle_birth_time")],
            )
            self.age = age.to("Gyr").value
        except KeyError:
            self.age = np.zeros(_n)

        try:
            self.metal = np.array(
                comp._yt_get_field(
                    self._ad,
                    [("bns", "particle_metallicity"), ("bns", "particle_metal")],
                ).value
            )
        except KeyError:
            self.metal = np.zeros(_n)

        try:
            self.Eu = np.array(
                comp._yt_get_field(
                    self._ad,
                    [("bns", "particle_bns_enrichment"), ("bns", "bns_enrichment")],
                ).value
            )
        except KeyError:
            self.Eu = np.zeros(_n)

        try:
            self.vkick1 = comp._yt_get_field(
                self._ad, [("bns", "particle_vkick1")]
            ).to("km/s").value
        except KeyError:
            self.vkick1 = np.zeros(_n)

        try:
            self.t_sn2 = np.array(
                comp._yt_get_field(self._ad, [("bns", "particle_t_sn2")]).value
            )
        except KeyError:
            self.t_sn2 = np.zeros(_n)

        try:
            self.vkick2 = comp._yt_get_field(
                self._ad, [("bns", "particle_vkick2")]
            ).to("km/s").value
        except KeyError:
            self.vkick2 = np.zeros(_n)

        try:
            self.t_merge = np.array(
                comp._yt_get_field(self._ad, [("bns", "particle_t_merge")]).value
            )
        except KeyError:
            self.t_merge = np.zeros(_n)

        try:
            self.parent_id = np.array(
                comp._yt_get_field(self._ad, [("bns", "particle_parent_id")]).value,
                dtype=np.int64,
            )
        except KeyError:
            self.parent_id = np.zeros(_n, dtype=np.int64)

        try:
            self.m1 = comp._yt_get_field(
                self._ad, [("bns", "particle_m1")]
            ).to("Msun").value
        except KeyError:
            self.m1 = np.zeros(_n)

    def halo_Only(self, center, n, r200, simple=False):
        super().halo_Only(center, n, r200, simple=simple)
        in_halo = np.where(self.r <= n * r200)
        self.stage   = self.stage[in_halo]
        self.age     = self.age[in_halo]
        self.metal   = self.metal[in_halo]
        self.Eu      = self.Eu[in_halo]
        self.vkick1    = self.vkick1[in_halo]
        self.t_sn2     = self.t_sn2[in_halo]
        self.vkick2    = self.vkick2[in_halo]
        self.t_merge   = self.t_merge[in_halo]
        self.parent_id = self.parent_id[in_halo]
        self.m1        = self.m1[in_halo]
        self.r         = self.r[in_halo]
