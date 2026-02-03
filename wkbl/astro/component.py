import __future__
import numpy as np
import sys
import yt
from . import nbody_essentials as nbe


def _yt_get_field(ad, candidates, default=None):
    for field in candidates:
        try:
            return ad[field]
        except Exception:
            continue
    if default is None:
        raise KeyError("None of the fields found: {}".format(candidates))
    return default

class Component:
    def __init__(self, file_path,comp,p, **kwargs):
        self._p = p
        self._center_history = np.array([0.,0.,0.])##########
        comov = kwargs.get('comov',False)
        self.halo_vel = kwargs.get('halo_vel',[0.,0.,0.])    ##########
        ds = kwargs.get("ds")
        ad = kwargs.get("ad")
        if ds is None or ad is None:
            ds = yt.load(file_path)
            ad = ds.all_data()
        self._ds = ds
        self._ad = ad
        if comp == "halo":
            ptype = "dark_matter"
            pos_x = _yt_get_field(ad, [(ptype, "particle_position_x")])
            pos_y = _yt_get_field(ad, [(ptype, "particle_position_y")])
            pos_z = _yt_get_field(ad, [(ptype, "particle_position_z")])
            vel_x = _yt_get_field(ad, [(ptype, "particle_velocity_x")])
            vel_y = _yt_get_field(ad, [(ptype, "particle_velocity_y")])
            vel_z = _yt_get_field(ad, [(ptype, "particle_velocity_z")])
            mass = _yt_get_field(ad, [(ptype, "particle_mass")])
            self.id = _yt_get_field(
                ad,
                [(ptype, "particle_index"), (ptype, "particle_id"),
                 (ptype, "particle_identifier")],
                default=np.arange(pos_x.size)
            )
        elif comp == "stars":
            ptype = "star"
            pos_x = _yt_get_field(ad, [(ptype, "particle_position_x")])
            pos_y = _yt_get_field(ad, [(ptype, "particle_position_y")])
            pos_z = _yt_get_field(ad, [(ptype, "particle_position_z")])
            vel_x = _yt_get_field(ad, [(ptype, "particle_velocity_x")])
            vel_y = _yt_get_field(ad, [(ptype, "particle_velocity_y")])
            vel_z = _yt_get_field(ad, [(ptype, "particle_velocity_z")])
            mass = _yt_get_field(ad, [(ptype, "particle_mass")])
            self.id = _yt_get_field(
                ad,
                [(ptype, "particle_index"), (ptype, "particle_id"),
                 (ptype, "particle_identifier")],
                default=np.arange(pos_x.size)
            )
        elif comp == "gas":
            pos_x = _yt_get_field(ad, [("index", "x"), ("gas", "x")])
            pos_y = _yt_get_field(ad, [("index", "y"), ("gas", "y")])
            pos_z = _yt_get_field(ad, [("index", "z"), ("gas", "z")])
            vel_x = _yt_get_field(ad, [("gas", "velocity_x")])
            vel_y = _yt_get_field(ad, [("gas", "velocity_y")])
            vel_z = _yt_get_field(ad, [("gas", "velocity_z")])
            try:
                mass = _yt_get_field(ad, [("gas", "cell_mass")])
            except KeyError:
                rho = _yt_get_field(ad, [("gas", "density")])
                vol = _yt_get_field(ad, [("index", "cell_volume")])
                mass = rho * vol
            self.id = np.arange(pos_x.size)
        else:
            raise ValueError("Unknown component: {}".format(comp))
        ### coordinates ###
        if comov:
            pos_x = pos_x.to("kpc").value / self._p.aexp
            pos_y = pos_y.to("kpc").value / self._p.aexp
            pos_z = pos_z.to("kpc").value / self._p.aexp
        else:
            pos_x = pos_x.to("kpc").value
            pos_y = pos_y.to("kpc").value
            pos_z = pos_z.to("kpc").value
        vel_x = vel_x.to("km/s").value
        vel_y = vel_y.to("km/s").value
        vel_z = vel_z.to("km/s").value
        self.pos3d = np.vstack((pos_x, pos_y, pos_z)).T
        self.vel3d = np.vstack((vel_x, vel_y, vel_z)).T
        self.mass = mass.to("Msun").value
        self.id = np.array(self.id)

    def halo_Only(self, center,n , r200,simple=False):
        ### particles ##
        self.r = np.sqrt((self.pos3d[:,0]**2)+(self.pos3d[:,1]**2)+(self.pos3d[:,2]**2))
        in_halo = np.where(self.r <= n*r200)
        in_r200= np.where(self.r <= r200)
        self.pos3d = self.pos3d[in_halo]
        self.mass = self.mass[in_halo]
        self.vel3d = self.vel3d[in_halo]
        self.id = self.id[in_halo]
        if not simple:
            r = self.r[in_halo]
            # spherical/cylindrical coordinates
            self.R = np.sqrt((self.pos3d[:,0]**2)+(self.pos3d[:,1]**2))
            self.phi = np.arctan2(np.copy(self.pos3d[:,1]),np.copy(self.pos3d[:,0]))
            self.theta = np.arccos(np.copy(self.pos3d[:,2])/np.copy(r))
            ### velocities ###
            vx,vy,vz = self.vel3d[:,0],self.vel3d[:,1],self.vel3d[:,2]
            self.v = np.sqrt((vx**2) + (vy**2) + (vz**2))
            self.vR = (vx*self.pos3d[:,0] + vy*self.pos3d[:,1])/ self.R
            self.vr = (vx*self.pos3d[:,0] + vy*self.pos3d[:,1] + vz*self.pos3d[:,2])/ r
            self.vphi = (-vx*self.pos3d[:,1] + vy*self.pos3d[:,0] )/ self.R
            self.vtheta = (self.vR*self.pos3d[:,2] - vz*self.R) / r
            #### other params ###
            self.total_m =  np.sum(self.mass[r<r200])

    def rotate(self,T):
        self.pos3d = nbe.matrix_vs_vector(T,self.pos3d)
        self.vel3d = nbe.matrix_vs_vector(T,self.vel3d)

    def vel_frame(self,vx_av,vy_av,vz_av):
        self.average_v = np.array([vx_av,vy_av,vz_av])
        self.vel3d = self.vel3d - self.average_v

    def shift(self,center):
        self.pos3d = self.pos3d - center
        self._center_history = np.vstack((self._center_history,center))

