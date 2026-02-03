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
        try:
            age = comp._yt_get_field(
                self._ad,
                [("star", "particle_age"), ("star", "age")],
            )
            self.age = age.to("Gyr").value
        except KeyError:
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
        
    def halo_Only(self, center, n, r200, simple=False):
        #### sf history ####
        if (self.gotsfInfo):
            self.sf_info.halo_Only(center, n, r200)
        super().halo_Only(center,n , r200, simple=simple)
        in_halo = np.where(self.r <= n*r200)
        self.age = self.age[in_halo]
        self.metal = self.metal[in_halo]
        self.r = self.r[in_halo]
 
    def shift(self,center):
        if (self.gotsfInfo):self.sf_info.shift(center)
        super().shift(center)
        
