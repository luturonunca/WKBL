import sys,os
import math
import glob
import cmath
#import cfalcon
import subprocess
import numpy as np
from unsio import *
import scipy.special as sp
from numpy import exp, sqrt
import nbody_essentials as nbe
import scipy.integrate as integrate
from sklearn.neighbors import KDTree
#CF =cfalcon.CFalcon()

class SF_info: 
    def __init__(self, file_path,p, **kwargs):
        if file_path[-1]=="/":
            file_path=file_path[:-1]
           
        # load st_info where the current stars where formed
        nufile_path = file_path[:-5]+"{0:05d}".format(int(file_path[-5:])-1)
        #star_files = glob.glob(file_path+"/star*")
        star_files = glob.glob(nufile_path+"/star*")
        
        if len(star_files)==0:
            # if there is no star files in the output directory
            # proceed to use the previous output, sometimes the 
            # very last output does not present any star files because
            # is just a final closing output
            num = float(nufile_path[-5:])
        self.p = p 
        comov= kwargs.get('comov',False)
        self._center_history = np.array([[0,0,0]])
        self.data = nbe._read_extra(nufile_path, sf_hist=True)
        stars = np.where(self.data[:,0]<=1)[0]
        if (comov):
            self.pos3d = self.data[stars,4:7] * p.simutokpc / p.aexp
        else:
            self.pos3d = self.data[stars,4:7] * p.simutokpc

        self.mass = self.data[stars,3] * p.simutoMsun
        self.kind = self.data[stars,0]
        self.id = self.data[stars,1]
        self.hsml = 25000./(2.**(self.data[stars,2]))
        self.rho = self.data[stars,10]*self.p.scale_d / self.p.scale_d_gas
        self.tokelvin = 1.66e-27 / (1.3806200e-19) * (self.p.unitl / self.p.unitt)**2
        self.temp = self.data[stars,14]/self.data[stars,10]* self.tokelvin
        self.met = self.data[stars,14] 
        self.size = self.p.boxlen/2**self.data[stars,2] 
        self.level = self.data[stars,2] 
        self.sigma2 = self.data[stars,10+self.p.nener] * (self.p.simutokms**2)
        cs2 = 0.666* self.temp * (self.p.simutokms**2) / self.tokelvin
        cs2[np.where(cs2<=0)] = 0.00002
        self.cs2 = cs2
        g_star = 1.6
        self.cs2_poly  = (self.p.nml['T2_star']/ self.tokelvin)
        self.cs2_poly *= (self.data[stars,10]*self.p.scale_nH/self.p.nml["n_star"])**(g_star-1.0)
        #self.cs2 = cs2 - (cs2_poly * self.p.simutokms**2)
        self.M2 = self.sigma2 / self.cs2
        factG = 3. / 4. / 2. / np.pi * 0.3089 * self.p.aexp 
        self.alpha0 = 0.5 * self.sigma2 / np.pi / factG / self.rho / (self.hsml)**2
        phi_x = 0.19
        self.rho_crit = ((np.pi**2)/5.) * (phi_x**2) * self.alpha0 * self.M2
         
    def halo_Only(self, center, n, r):
        self.r = np.sqrt((self.pos3d[:,0]**2)+(self.pos3d[:,1]**2)+(self.pos3d[:,2]**2))
        in_halo = np.where(self.r <= n*r)
        self.pos3d = self.pos3d[in_halo] - center
        self.kind = self.kind[in_halo]
        self.id = self.id[in_halo]
        self.rho = self.rho[in_halo]
        self.size = self.size[in_halo]
        self.level = self.level[in_halo]
        self.rho_crit = self.rho_crit[in_halo]
        self.temp = self.temp[in_halo]
        self.met = self.met[in_halo]
        self.alpha0= self.alpha0[in_halo]
        self.sigma2 = self.sigma2[in_halo]
        self.cs2 = self.cs2[in_halo]
        self.M2 = self.M2[in_halo]
        self.data = self.data[in_halo]
        self.r = self.r[in_halo]

    
    def rotate(self,T):
        pos = self.pos3d
        self.pos3d = nbe.matrix_vs_vector(T,pos) 
   
    def id_sort(self):
        sorted_by_id = np.argsort(self.id)
        self.pos3d = self.pos3d[sorted_by_id]
        self.id = self.id[sorted_by_id]
        self.rho = self.rho[sorted_by_id]
        self.rho_crit = self.rho_crit[sorted_by_id]
        self.temp2 = self.temp2[sorted_by_id]
        self.met = self.met[sorted_by_id]
        self.alpha0= self.alpha0[sorted_by_id]
        self.sigma2 = self.sigma2[sorted_by_id]
        self.cs2 = self.cs2[sorted_by_id]
        self.M2 = self.M2[sorted_by_id]    
    
  
    def shift(self, center):
        self.pos3d = self.pos3d - center
        self._center_history = np.vstack((self._center_history,center))

