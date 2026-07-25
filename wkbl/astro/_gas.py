from . import component as comp
import numpy as np

class _gas(comp.Component):
    def __init__(self, file_path,p,dens=True, **kwargs):
        self.get_sigma = kwargs.get('virial',p.nmlexist)
        comov = kwargs.get('comov',False)
        ds = kwargs.get("ds")
        ad = kwargs.get("ad")
        super().__init__(file_path, "gas", p, comov=comov, ds=ds, ad=ad)
        # Clear yt cache after pos/vel/mass are in numpy — each subsequent
        # field read re-uses _ad but the cache is freed immediately after.
        self._ad.clear_data()

        def _get(candidates, unit=None):
            try:
                _f = comp._yt_get_field(self._ad, candidates)
                result = _f.to(unit).value if unit else np.array(_f.value)
                del _f
                self._ad.clear_data()
                return result
            except KeyError:
                return None

        _n = len(self.mass)

        _r = _get([("gas", "density")], "Msun/kpc**3")
        self.rho = _r if _r is not None else np.zeros(_n)

        _t = _get([("gas", "temperature")], "K")
        self.temp = _t if _t is not None else np.zeros(_n)

        _pot = _get([("gas", "potential")])
        self.pot = _pot if _pot is not None else np.zeros(_n)

        try:
            _pf = comp._yt_get_field(
                self._ad, [("gas", "pressure"), ("gas", "thermal_pressure")]
            )
            try:
                self.pres = np.array(_pf.to("code_pressure").value)
            except Exception:
                self.pres = np.array(_pf.value)
            del _pf
            self._ad.clear_data()
        except KeyError:
            self.pres = np.zeros(_n)

        _m = _get([("gas", "metallicity"), ("gas", "metal")])
        self.met = _m if _m is not None else np.zeros(_n)

        _eu = _get([("gas", "Eu"), ("ramses", "hydro_Eu")])
        self.Eu = _eu if _eu is not None else np.zeros(_n)

        _fe = _get([("gas", "Fe"), ("ramses", "hydro_Fe")])
        self.Fe = _fe if _fe is not None else np.zeros(_n)

        _mg = _get([("gas", "Mg"), ("ramses", "hydro_Mg")])
        self.Mg = _mg if _mg is not None else np.zeros(_n)

        try:
            _cv = comp._yt_get_field(self._ad, [("index", "cell_volume")])
            hsml = _cv.to("kpc**3").value ** (1.0 / 3.0)
            del _cv
            self._ad.clear_data()
        except KeyError:
            hsml = np.zeros(_n)
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
                _sig = comp._yt_get_field(
                    self._ad, [("gas", "velocity_dispersion")]
                ).to("km/s").value
                self._ad.clear_data()
                self.sigma2 = _sig**2
            except KeyError:
                self.sigma2 = np.zeros(_n)
            self.cs2 = (1.6667-1.) * self.pres * (self._p.simutokms**2)
            g_star = 1.6
            
    def halo_Only(self, center,n , r200,simple=False):
        super().halo_Only(center,n , r200,simple=simple)
        in_halo = np.where(self.r <= n*r200)
        self.met = self.met[in_halo]
        self.Eu = self.Eu[in_halo]
        self.Fe = self.Fe[in_halo]
        self.Mg = self.Mg[in_halo]
        self.pot = self.pot[in_halo]
        self.temp = self.temp[in_halo]
        self.pres = self.pres[in_halo]
        self.hsml = self.hsml[in_halo]
        self.rho = self.rho[in_halo]
        if (self.get_sigma):
            self.sigma2 = self.sigma2[in_halo]
            self.cs2 = self.cs2[in_halo]
        self.r = self.r[in_halo]
        
