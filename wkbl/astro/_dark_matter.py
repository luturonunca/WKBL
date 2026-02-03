from . import component as comp
from . import _clumps as clumps 
import numpy as np


class _dark_matter(comp.Component):
    def __init__(self, file_path,p, **kwargs):
        self.file = file_path
        hsml = kwargs.get('hsml',False)
        comov = kwargs.get('comov',False)
        rs = kwargs.get('rs',"")
        ds = kwargs.get("ds")
        ad = kwargs.get("ad")
        try:
            self.Clumps = clumps.Clumps(file_path,p,comov=comov,
                      rs=rs)
            self.subhalos = True
        except:
            self.subhalos = False
        super().__init__(file_path, "halo", p, comov=comov, ds=ds, ad=ad)
        #super().__init__()
    
    def halo_Only(self, center,n , r200,simple=False):
        #### clumps ###
        if (self.subhalos):self.Clumps.halo_Only(center, n, r200)
        super().halo_Only(center, n, r200, simple=simple)
        in_halo = np.where(self.r <= n*r200)
        self.r = self.r[in_halo]
    
    def rotate(self,T):
        if (self.subhalos):self.Clumps.rotate(T)
        super().rotate(T)
        
    def shift(self,center):
        if (self.subhalos):self.Clumps.shift(center)
        super().shift(center)
        
    
    def density_profile(self, bins, limit):
        r_p = np.logspace(-0.5, np.log10(limit),bins)
        def sph_dens(r):
            """
            spherical density profile
            """
            total_mass = np.sum(self.mass[(self.r < r)])
            dens = 4 * total_mass / 3. / np.pi / r**3
            return dens

        get_shp_dens = np.vectorize(sph_dens)
        return r_p , get_shp_dens(r_p)

        
