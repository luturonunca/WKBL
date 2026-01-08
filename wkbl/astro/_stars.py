from . import component as comp
from . import _sf_info as sfi
import numpy as np

class _stars(comp.Component):
    def __init__(self, file_path,p, **kwargs):
        dens = kwargs.get('dens',False)
        comov = kwargs.get('comov',False)
        r_search = kwargs.get('r_search',200.)
        try:
            self.sf_info = sfi.SF_info(file_path,p,comov=comov)
            self.gotsfInfo = True
        except:
            self.gotsfInfo = False
        super().__init__(file_path,"stars",p,comov=comov)
        ok, age = self.uns.getData("stars","age")
        ok, self.metal = self.uns.getData("stars","metal")
        ok, self.id = self.uns.getData("all","id")
        self.age = age *self._p.unitt / (3600.*24.*365*1e9) / self._p.aexp**2 # stars age to Gyrs
        
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
        
