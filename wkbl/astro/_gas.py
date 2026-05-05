from . import component as comp
import numpy as np

class _gas(comp.Component):
    def __init__(self, file_path,p,dens=True, **kwargs):
        self.get_sigma = kwargs.get('virial',p.nmlexist)
        comov = kwargs.get('comov',False)
        ds = kwargs.get("ds")
        ad = kwargs.get("ad")
        super().__init__(file_path, "gas", p, comov=comov, ds=ds, ad=ad)
        try:
            rho = comp._yt_get_field(self._ad, [("gas", "density")])
            self.rho = rho.to("Msun/kpc**3").value
        except KeyError:
            self.rho = np.zeros(len(self.mass))
        try:
            temp = comp._yt_get_field(self._ad, [("gas", "temperature")])
            self.temp = temp.to("K").value
        except KeyError:
            self.temp = np.zeros(len(self.mass))
        try:
            self.pot = np.array(
                comp._yt_get_field(self._ad, [("gas", "potential")]).value
            )
        except KeyError:
            self.pot = np.zeros(len(self.mass))
        try:
            acc_x = comp._yt_get_field(self._ad, [("gas", "acceleration_x")]).value
            acc_y = comp._yt_get_field(self._ad, [("gas", "acceleration_y")]).value
            acc_z = comp._yt_get_field(self._ad, [("gas", "acceleration_z")]).value
            self.acc = np.vstack((acc_x, acc_y, acc_z)).T
        except KeyError:
            self.acc = np.zeros((len(self.mass), 3))
        try:
            pres = comp._yt_get_field(
                self._ad,
                [("gas", "pressure"), ("gas", "thermal_pressure")],
            )
            try:
                self.pres = pres.to("code_pressure").value
            except Exception:
                self.pres = pres.value
        except KeyError:
            self.pres = np.zeros(len(self.mass))
        self.pres = np.array(self.pres)
        try:
            self.met = comp._yt_get_field(
                self._ad,
                [("gas", "metallicity"), ("gas", "metal")],
            ).value
        except KeyError:
            self.met = np.zeros(len(self.mass))
        self.met = np.array(self.met)
        try:
            self.Eu = comp._yt_get_field(
                self._ad,
                [("gas", "bns_enrichment"), ("ramses", "hydro_bns_enrichment")],
            ).value
        except KeyError:
            self.Eu = np.zeros(len(self.mass))
        self.Eu = np.array(self.Eu)
        if len(self.pres) == len(self.rho) and len(self.rho) > 0:
            temp2 = self.pres / self.rho
        #= self.pres/rho
        try:
            cell_vol = comp._yt_get_field(self._ad, [("index", "cell_volume")])
            hsml = cell_vol.to("kpc**3").value ** (1.0 / 3.0)
        except KeyError:
            hsml = np.zeros(len(self.mass))
        if comov:
            hsml /= self._p.aexp
        self.hsml = hsml
        if self._p.SHIFT:
            shift1 = (np.random.rand(len(hsml))-0.5)*self.hsml
            shift2 = (np.random.rand(len(hsml))-0.5)*self.hsml
            shift3 = (np.random.rand(len(hsml))-0.5)*self.hsml
            self.pos3d[:,0]+=shift1
            self.pos3d[:,1]+=shift2
            self.pos3d[:,2]+=shift3
        self.tokelvin = self._p.mH / (1.3806200e-16) * (self._p.unitl / self._p.unitt)**2
        if (self.get_sigma):
            try:
                sigma = comp._yt_get_field(
                    self._ad, [("gas", "velocity_dispersion")]
                ).to("km/s").value
                self.sigma2 = sigma**2
            except KeyError:
                self.sigma2 = np.zeros(len(self.mass))
            self.cs2 = (1.6667-1.) * self.pres * (self._p.simutokms**2)
            g_star = 1.6
            
    def halo_Only(self, center,n , r200,simple=False):
        super().halo_Only(center,n , r200,simple=simple)
        in_halo = np.where(self.r <= n*r200)
        self.met = self.met[in_halo]
        self.Eu = self.Eu[in_halo]
        self.pot = self.pot[in_halo]
        self.temp = self.temp[in_halo]
        self.pres = self.pres[in_halo]
        self.hsml = self.hsml[in_halo]
        self.rho = self.rho[in_halo]
        if (self.get_sigma):
            self.sigma2 = self.sigma2[in_halo]
            self.cs2 = self.cs2[in_halo]
        self.r = self.r[in_halo]
        
