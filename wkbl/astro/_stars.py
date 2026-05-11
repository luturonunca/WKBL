from . import component as comp
from . import _sf_info as sfi
import numpy as np

class _stars(comp.Component):
    def __init__(self, file_path,p, **kwargs):
        dens = kwargs.get('dens',False)
        comov = kwargs.get('comov',False)
        r_search = kwargs.get('r_search',200.)
        ds = kwargs.get("ds")
        ad = kwargs.get("ad")
        try:
            self.sf_info = sfi.SF_info(file_path,p,comov=comov)
            self.gotsfInfo = True
        except:
            self.gotsfInfo = False
        super().__init__(file_path, "stars", p, comov=comov, ds=ds, ad=ad)
        _SEC_PER_GYR = 1e9 * 365.25 * 24.0 * 3600.0
        try:
            _tau_birth = comp._yt_get_field(
                self._ad, [("star", "conformal_birth_time")]
            ).value
            _use_proper = getattr(p, 'nml', {}).get("use_proper_time", False)
            if _use_proper:
                # tau_birth is negative lookback proper time; age = lookback time to birth
                self.age = -_tau_birth * p.unitt / p.aexp**2 / _SEC_PER_GYR
            else:
                self.age = (p.time - _tau_birth) * p.unitt / p.aexp**2 / _SEC_PER_GYR
        except Exception:
            self.age = np.zeros(len(self.mass))
        try:
            self.metal = comp._yt_get_field(
                self._ad,
                [("star", "particle_metallicity"), ("star", "metallicity"),
                 ("star", "metal")],
            )
        except KeyError:
            self.metal = np.zeros(len(self.mass))
        self.metal = np.array(self.metal)
        try:
            self.Eu = comp._yt_get_field(
                self._ad,
                [("star", "particle_bns_enrichment"), ("star", "bns_enrichment")],
            )
        except KeyError:
            self.Eu = np.zeros(len(self.mass))
        self.Eu = np.array(self.Eu)

    def halo_Only(self, center, n, r200, simple=False):
        #### sf history ####
        if (self.gotsfInfo):
            self.sf_info.halo_Only(center, n, r200)
        super().halo_Only(center,n , r200, simple=simple)
        in_halo = np.where(self.r <= n*r200)
        self.age = self.age[in_halo]
        self.metal = self.metal[in_halo]
        self.Eu = self.Eu[in_halo]
        self.r = self.r[in_halo]
 
    def shift(self,center):
        if (self.gotsfInfo):self.sf_info.shift(center)
        super().shift(center)
        
