import  __future__
import numpy as np
import sys
from unsio.input import CUNS_IN as CunsIn
from . import nbody_essentials as nbe

class Component:
    def __init__(self, file_path,comp,p, **kwargs):
        self._p = p
        self._center_history = np.array([0.,0.,0.])##########
        self.uns = CunsIn(file_path,comp)
        comov = kwargs.get('comov',False)
        self.halo_vel = kwargs.get('halo_vel',[0.,0.,0.])    ##########
        ok = self.uns.nextFrame("")
        ok, pos = self.uns.getData(comp,"pos")
        ok, vel = self.uns.getData(comp,"vel")
        ok, mass = self.uns.getData(comp,"mass")
        ok, self.id = self.uns.getData(comp,"id")
        ### coordinates ###
        if (comov):
            pos = pos * self._p.simutokpc / self._p.aexp
        else:
            pos = pos * self._p.simutokpc
        vel = vel * self._p.simutokms
        self.pos3d = pos.reshape(int(len(pos)/3),3)
        self.vel3d = vel.reshape(int(len(vel)/3),3)
        self.mass = mass * self._p.simutoMsun

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



