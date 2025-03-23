from . import component as comp
import numpy as np

class _gas(comp.Component):
    def __init__(self, file_path,p,dens=True, **kwargs):
        self.get_sigma = kwargs.get('virial',p.nmlexist)
        comov = kwargs.get('comov',False)
        super().__init__(file_path,"gas",p,comov=comov)
        ok, temp = self.uns.getData("gas","temp")
        ok, rho = self.uns.getData("gas","rho")
        ok, pot = self.uns.getData("gas","pot")
        ok, acc = self.uns.getData("gas","acc")
        ok, self.pres = self.uns.getData("hydro","4")
        ok, self.met = self.uns.getData("hydro","5")
        temp2 = self.pres/rho
        #= self.pres/rho
        self.rho =  rho * self._p.simutoMsun / (self._p.simutokpc**3)
        self.pot = pot * self._p.unitm * (self._p.unitl/self._p.unitt)**2
        self.acc = acc * self._p.unitl / self._p.unitt**2
        ok, hsml = self.uns.getData("gas","hsml")
        if (comov):
            hsml/=self._p.aexp
        self.hsml = hsml * self._p.simutokpc
        if self._p.SHIFT:
            shift1 = (np.random.rand(len(hsml))-0.5)*self.hsml
            shift2 = (np.random.rand(len(hsml))-0.5)*self.hsml
            shift3 = (np.random.rand(len(hsml))-0.5)*self.hsml
            self.pos3d[:,0]+=shift1
            self.pos3d[:,1]+=shift2
            self.pos3d[:,2]+=shift3
        self.tokelvin = self._p.mH / (1.3806200e-16) * (self._p.unitl / self._p.unitt)**2
        self.temp = temp * self.tokelvin
        if (self.get_sigma):
            ok, sigma = self.uns.getData("hydro",str(self._p.nener))
            self.sigma2 = sigma*(self._p.simutokms**2)
            self.cs2 = (1.6667-1.) * self.pres * (self._p.simutokms**2)
            g_star = 1.6
            
    def halo_Only(self, center,n , r200,simple=False):
        super().halo_Only(center,n , r200,simple=simple)
        in_halo = np.where(self.r <= n*r200)
        self.met = self.met[in_halo]
        self.pot = self.pot[in_halo]
        self.temp = self.temp[in_halo]
        self.pres = self.pres[in_halo]
        self.hsml = self.hsml[in_halo]
        self.rho = self.rho[in_halo]
        if (self.get_sigma):
            self.sigma2 = self.sigma2[in_halo]
            self.cs2 = self.cs2[in_halo]
        self.r = self.r[in_halo]
        
