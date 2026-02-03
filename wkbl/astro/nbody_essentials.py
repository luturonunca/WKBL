import os,sys,re
import math
import glob
import cmath
#from iminuit import Minuit
import subprocess
import numpy as np
import scipy.special as sp
from numpy import exp, sqrt
import scipy.integrate as integrate
from sklearn.neighbors import KDTree


class Info_sniffer:
    def __init__(self, file_path,newage=False):
        _vars=dict()
        self.file_path = file_path
        file = glob.glob(file_path+"/info_?????.txt")[0]
        with open(file) as outinfo:
            for line in outinfo:
                eq_index = line.find('=')
                if eq_index == -1:
                    continue
                var_name = line[:eq_index].strip()
                number = float(line[eq_index + 1:].strip())
                _vars[var_name] = number
                if var_name == "unit_t":
                    break
        self.nmlexist = os.path.isfile(file_path+"/namelist.txt")
        if (self.nmlexist):
            nml = dict()

            with open(file_path+"/namelist.txt") as outinfo:
                for line in outinfo:
                    eq_index = line.find('=')
                    end_index = line.find('!')
                    if eq_index == -1:
                        continue
                    var_name = line[:eq_index].strip()
                    try:
                        val = float(line[eq_index + 1:end_index])
                    except:
                        if  line[eq_index+1:-1]=='.true.':
                            val = True
                        elif line[eq_index+1:-1]=='.false.':
                            val = False
                        else:
                            val = line[eq_index+1:-1]
                    nml[var_name] = val
            if not "levelmax" in nml:
                self.nmlexist = False
            nener=4
            if "metal" in nml:
                nener+= int(nml["metal"])
            if "delayed_cooling" in nml:
                nener += int(nml["delayed_cooling"])
            if "sf_virial" in nml:
                nener += int(nml["sf_virial"])
            self.nener = nener
            self.nml = nml
        
        self.H0 = _vars["H0"]
        self._vars = _vars
        self.h = _vars["H0"]/1e2       # hubble expansion rate
        self.aexp = _vars["aexp"]      # expantion parameter
        self.Z = -1. + (1./ self.aexp) # redshift 
        self.msuntokg = 1.99844e30   
        self.pctocm = 3.08567758e18 #but well
        self.G = 6.67384e-11 * self.msuntokg / ((self.pctocm*10)**3)#kpc^3 Msun^-1 s^-2
        #self.rho_crit = (3 * (self.H0**2) / 3.08567758e19**2)/ (8*np.pi*self.G)
        # rho crit in terms of aexp
        rho_crit = get_rho_crit(self.aexp, self.H0, _vars["omega_m"], _vars["omega_l"], self.G)
        self.rho_crit = rho_crit / 3.08567758e19**2 
        self.cmtopc = 1./self.pctocm
        self.unitl  = _vars["unit_l"]
        self.unitd  = _vars["unit_d"]
        self.unitt  = _vars["unit_t"]
        self.unitv  = self.unitl/self.unitt
        self.unitm  = self.unitd *  self.unitl**3
        self.boxlen = self.unitl/self.pctocm/1e6 #Mpc
        self.simutokpc  = self.unitl/self.pctocm/1e3
        self.simutokms  = self.unitv/1e5
        self.simutoMsun = self.unitm/1e3/self.msuntokg
        self.unitsimutoMsunkpc3=self.unitd*self.pctocm**3/1000/self.msuntokg
        self.kgtoGeV = 1/1.783e-27
        self.kpctocm = 3.086e21
        self.simutocm=self.unitl*_vars["H0"]/1e2
        self.kpctokm = self.kpctocm / 1e5
        self.simutoGeVcm3 = (self.simutoMsun*self.msuntokg*self.kgtoGeV) / (self.simutokpc*self.kpctocm)**3
        self.kB = 1.3806503e-23 * (self.cmtopc/ 10)**2 / self.msuntokg # kpc**2 Msun /s**2 / K
        self.k_boltz = 1.3806488e-23 * 1e-6 / self.msuntokg # Msun * km**2 /s**2 / K
        self.mu = 1.67262158e-27 / self.msuntokg
        self.mH = 1.6600000e-24 # grams
        self.kB = 1.3806200e-16 # cgs
        self.scale_T2 =  self.mH/self.kB * (self.simutocm/self.unitt)**2 #simutokelvin
        self.scale_d = self.simutoMsun * (self.simutokpc)**-3
        self.scale_nH = 0.76 * self.scale_d / self.mH
        self.omegaM = 0.0489
        self.scale_d_gas = self.omegaM*self.rho_crit*((80./100)**2)/(self.aexp**3)
        scale_nH = self.unitd / 1.66e-24 * 0.76 # scale_d * X / mh
        if (self.nmlexist):
            self.reslim = self.boxlen*1e3/2**(self.nml["levelmax"])
        """
        f = open(self.file_path+"/namelist.txt")
        for l in f:
            row = l.split('=')
            if row[0]=="sf_model":
                SF0 = not bool(int(row[1]))
            if row[0]=="n_star":
                n_star=float(row[1])
                self.n_star = n_star
        """
        try:
            self.rhoc_SF = n_star * (self.mH * 1e-3)/(self.msuntokg) / ((self.cmtopc/1e3)**3) #Msun /kpc^3
            self.DelayedCooling=True
        except:
            self.DelayedCooling=False


class Profile:
    def __init__(self,simu,hsml,powerR =False,stars=False,salve=False,fixgama=False,**kwargs):
        try:
            mmm = np.min(simu.dm.rho)
        except ValueError: 
            print("the density field is ill defined!")
        self.hsml = hsml
        transition = kwargs.get('transition',3)
        fit = kwargs.get('fit',False)
        quiet = kwargs.get('quiet',0)
        fixalpha = kwargs.get('fixalpha',False)
        alpha = kwargs.get('alpha',1)
        Mdm = simu.dm.mass.min()
        myradiuses = simu.dm.r[np.argsort(simu.dm.r)]
        tabN = np.cumsum(np.ones(len(myradiuses)))[1:]
        myradiuses = myradiuses[1:]
        Pcrit = simu.dm._p.rho_crit
        Rp03 = np.sqrt(200/64.) * np.sqrt(4 * np.pi * Pcrit * tabN / 3. / Mdm ) * (myradiuses**1.5)/ np.log(tabN) 
        val =0.6
        self.R_P03 = myradiuses[ np.where(Rp03 > val) ][0]
        
        # R array logarithmic Bining
        r_p = np.logspace(np.log10(0.035),np.log10(transition*self.hsml),25)
        if (powerR): 
            self.hsml=self.R_P03
            r_p = np.logspace(np.log10(0.15),np.log10(self.R_P03),25)
            if  (stars):
                r_p = np.logspace(np.log10(0.035),np.log10(self.R_P03),25)

        # histogram of dm particles per logarithmic bin
        n_dm,r = np.histogram(simu.dm.r,bins=r_p)
        # edges of bins
        r1,r2 =r[:-1],r[1:]
        # shell's volume
        vol = 4.* np.pi * ((r2**3)-(r1**3)) / 3.
        r_size = r_p[1:]-r_p[:-1]
        # density per shell
        self.profile_in = n_dm*simu.dm.mass.min()/vol
        self.cummass_in = np.cumsum(n_dm)*simu.dm.mass.min()
        # center of bins
        self.r_in = 10**((np.log10(r_p[:-1])+np.log10(r_p[1:]))/2.)


        # R array logarithmic Bining
        r_p = np.logspace(np.log10(self.R_P03),np.log10(2.5*simu.r200),150)
        if not (powerR):r_p = np.logspace(np.log10(transition*self.hsml),np.log10(2.5*simu.r200),150)
            
        # histogram of dm particles per logarithmic bin
        n_dm,r = np.histogram(simu.dm.r,bins=r_p)
        # edges of bins
        r1,r2 =r[:-1],r[1:]
        # shell's volume
        vol = 4.* np.pi * ((r2**3)-(r1**3)) / 3.
        r_size = r_p[1:]-r_p[:-1]
        # density per shell
        self.profile = n_dm*simu.dm.mass.min()/vol
        self.cummass = np.cumsum(n_dm)*simu.dm.mass.min()
        # center of bins
        self.r = 10**((np.log10(r_p[:-1])+np.log10(r_p[1:]))/2.)
        bin_size= (r_p[:-1]-r_p[1:])/2.
        
        if (stars):
            r_p_st = np.logspace(np.log10(self.hsml),np.log10(2.5*simu.r200),150)
            # histogram of dm particles per logarithmic bin
            n_st,r = np.histogram(simu.st.r,bins=r_p_st)
            mass_st,r = np.histogram(simu.st.r,bins=r_p_st,weights=simu.st.mass)

            # edges of bins
            r1,r2 =r[:-1],r[1:]
            # shell's volume
            vol = 4.* np.pi * ((r2**3)-(r1**3)) / 3.
            r_size = r_p_st[1:]-r_p_st[:-1]
            # density per shell
            self.profile_st = mass_st/vol
            self.cummass_st = np.cumsum(mass_st)
            # center of bins
            self.r_st = 10**((np.log10(r_p_st[:-1])+np.log10(r_p_st[1:]))/2.)
            bin_size= (r_p_st[:-1]-r_p_st[1:])/2.    

        # extra estatistics from Cfalcon density
        mean = std = n = stdlog = np.array([])
        mean_st = std_st = n_st = np.array([])
        for i in range(len(r_p)-1):
            shell = np.where((simu.dm.r > r_p[i])&(simu.dm.r < r_p[i+1])&(simu.dm.r > hsml))
            n = np.append(n,len(shell[0]))
            mean = np.append(mean,np.mean(simu.dm.rho[shell]))
            std = np.append(std,np.std(simu.dm.rho[shell]))
            stdlog = np.append(stdlog,np.std(np.log10(simu.dm.rho[shell])))
            if (stars) and (salve):
                # stupid fix due to Cfalcon crash
                shell_st = np.where((simu.st.r[res_attempt] > r_p_st[i])&(simu.st.r[res_attempt] < r_p_st[i+1]))
                mean_st = np.append(mean_st,np.mean(simu.st.rho[shell_st]))
                std_st = np.append(std_st,np.std(simu.st.rho[shell_st]))
                n_st = np.append(n_st,len(shell_st[0]))
            elif (stars):
                shell_st = np.where((simu.st.r > r_p_st[i])&(simu.st.r < r_p_st[i+1]))
                
                mean_st = np.append(mean_st,np.mean(simu.st.rho[shell_st]))
                std_st = np.append(std_st,np.std(simu.st.rho[shell_st]))
                n_st = np.append(n_st,len(shell_st[0]))

        self.mean = mean
        self.std = std
        self.stdlog = stdlog
        self.n_dm_bin = n
        if (stars):
            self.mean_st = mean_st
            self.std_st = std_st
            self.n_dm_bin_st = n_st
        n = np.array([len(simu.dm.mass[simu.dm.r<i]) for i in r]) 
        if (fit):
            self.m_rho = Minuit(self.chi2_rho_log,print_level=quiet,
                           ga=1., fix_ga=fixgama,fix_al=fixalpha,
                           po=7.0,    error_po=0.1,  limit_po =(2.,11.),
                           r_s=7.3,  error_r_s=0.1,   limit_r_s=(5.,80),
                           be=3.,     error_be=0.1,   limit_be =(2.5,3.5),
                           al=alpha,     error_al=0.1,   limit_al =(.5,1.5))
            self.m_rho.migrad();        
        
    def chi2_rho_log(self,po,r_s,al,be,ga):
        """
        logarithmic Chi-square
        using mean of rho per shell
        """
        rho_obs = self.profile
        rho_the = np.array([abg_profile(i,po,r_s,al,be,ga) for i in self.r])
        c = (np.log10(rho_the) - np.log10(rho_obs))/ self.stdlog
        c = c**2
        return np.sum(c)


def abg_profile(x,po,r_s,al,be,ga):
    power =  (be - ga) / al
    denominator = ((x/r_s)**ga) * ((1 + (x / r_s)**al)**power)
    return (10**po) / denominator


def FIRE_st_mass(mass,r,r97):
    """
    from arxiv:1702.06148v1 footnote 9
    returns the mass defined there as M_st
    and the radius they define
    to plot on top of the Moster band
    """
    R15 = 0.15*r97
    R_half = half_mass(mass[r< R15],r[r< R15])
    R_FIRE = half_mass(mass[r< 3*R_half],r[r< 3*R_half])
    return mass[r < R_FIRE].sum(), R_FIRE
    

def Delta_crit(a, omg_m, omg_l):
    """
    Delta crit a la Brian and Norman 1998
    """
    x = (omg_m/(omg_m+(a**3)*omg_l))-1
    return (18.*np.pi**2) + (82.*x) - (39.* x**2)

def a_dot(a,h0,omg_m, omg_l):
    """
    time derivative of the expantion paramiter
    """
    omg_k = 1- omg_m - omg_l
    return h0 * a * np.sqrt((omg_m*a**(-3))+(omg_k*a**(-2))+omg_l)

def get_rho_crit(a, h0, omg_m, omg_l, G):
    """
    time dependent critical density of the universe
    """
    H_z = a_dot(a,h0,omg_m, omg_l) / a
    return 3. * H_z**2 /8. / np.pi / G 

def fix_fortran_numbers(line):
    # Fix cases like "0.123456789012-130" → "0.123456789012E-130"
    return re.sub(r'(\d)([+-]\d{3})', r'\1E\2', line)


def _read_extra(output,path="",clumps=False,rockstar=False,sf_hist=False):
        """
        previously named _get_center()
        gets center of more resolved halo
        taken from hast library witten by V.perret
        https://bitbucket.org/vperret/hast 
        """
        p = Info_sniffer(output)
        data_all = np.array([])
        if (sf_hist):
            list = glob.glob(output+'/stars_?????.out?????')
        elif (clumps):
            list = glob.glob(output+'/clump_?????.txt?????')
        if (clumps)and(rockstar):
            h = p.h
            files = glob.glob(path+"/halos*.ascii")
            i=0
            for catalog in files:
                if i==0:
                    data_all = np.loadtxt(catalog)
                else:
                    data_all = np.vstack((data_all,np.loadtxt(catalog)))
                i+=1
            return data_all
        else:
            i=0
            for file in list:
                    if os.path.getsize(file)==183:continue
                    try:
                        #data = np.loadtxt(file,skiprows=1,dtype=None)
                        data = np.loadtxt(file,dtype=None)
                    except:
                        try:
                            with open(file) as f:
                                fixed_lines = [fix_fortran_numbers(line) for line in f]
                            data = np.loadtxt(fixed_lines)
                        except:
                            continue
                    if(np.size(data)==0):
                            continue
                    if(i>0):
                            data_all = np.vstack((data_all,data))
                    else:
                            data_all = data
                    i=i+1
            if not bool(len(data_all)): 
                print("no stars log")
                return np.array([])
            indicator = len(np.shape(data_all))
            if indicator==1:
                array=(1e4*data_all[3]/np.max(data_all[3]))*(data_all[8]/np.max(data_all[8])).astype(int, copy=False)
            else:
                array=(1e4*data_all[:,3]/np.max(data_all[:,3]))*(data_all[:,8]/np.max(data_all[:,8])).astype(int, copy=False)
            data_sorted = data_all[array.argsort()]
            data_sorted = data_sorted[::-1]
            if (sf_hist):
                return data_all
            elif not (clumps):
                return data_sorted[0,4:7]
            else:
                return data_sorted

def get_com(pos,m):
    return np.array([np.dot(pos[:,0],m),
                     np.dot(pos[:,1],m),
                     np.dot(pos[:,2],m)])/ np.sum(m)
def get_r(pos):
    return np.sqrt(pos[:,1]**2 + pos[:,1]**2 + pos[:,2]**2)

def get_radii(r,masses,p,r_max,bins=1080,rhoref=0):
        try:
            r_min = p.reslim # resolution-limit in kpc
        except:
            r_min = 0.5 
        mhist, rhist = np.histogram(r,range=(r_min,r_max),bins=bins, weights=masses )
        vol_bin = (4./3.)*np.pi*(rhist[1:]**3)
        r_bin = rhist[:-1]+ 0.5*(rhist[2]-rhist[1])
        rho_s = np.cumsum(mhist) / vol_bin
        delta_crit = Delta_crit(p.aexp,p._vars["omega_m"],p._vars["omega_l"])
        r200 = r_bin[np.argmin(np.abs(rho_s - (200 * p.rho_crit)))]
        r97  = r_bin[np.argmin(np.abs(rho_s - (97 * p.rho_crit)))]
        rBN  = r_bin[np.argmin(np.abs(rho_s - (delta_crit * p.rho_crit)))]
        rREF = r_bin[np.argmin(np.abs(rho_s - rhoref))]
        if rhoref>0:return delta_crit, r200,r97,rBN,rREF, 
        
        return delta_crit, r200,r97,rBN


def print_matrix(D):
    print('| Diagonal matrix computed ')
    print('|    | {0}, {1}, {2}|'.format(int(D[0,0]),int(D[0,1]),int(D[0,2])))
    print('| D =| {0}, {1}, {2}|'.format(int(D[1,0]),int(D[1,1]),int(D[1,2])))
    print('|    | {0},  {1}, {2}|'.format(int(D[2,0]),int(D[2,1]),int(D[2,2])))



def real_center(pos, mass, n=7000):
    """
    this method computes the center of mass of a recursively reducing
    sphere center in the previous COM as descrived by Schaller et al. 
    in
    arxiv:1505.05470v2
    input: dm, st, gas object from Galaxy_hound()
    coordinates of the real center of mass
 
    """
    p = np.copy(pos)
    m = np.copy(mass)
    final = np.zeros((1,3))
    while len(p) > n:
        com = get_com(p,m)
        final += com
        p -= com
        r = get_r(p)
        m = m[np.where(r<0.9*r.max())]
        p = p[np.where(r<0.9*r.max())]
    return final[0]

def matrix_vs_vector(mat,vec):
    """
    costume matrix vector multiplication to speed up the product between a matrix
    rotation matrix for example, and an array of vectors
    mat --> matrix
    vec --> array of 3d vectors
    output: the resulting array of 3d vectors
    """
    res = np.zeros((len(vec),3))
    res[:,0] = mat[0][0]*vec[:,0] + mat[0][1]*vec[:,1] + mat[0][2]*vec[:,2]
    res[:,1] = mat[1][0]*vec[:,0] + mat[1][1]*vec[:,1] + mat[1][2]*vec[:,2]
    res[:,2] = mat[2][0]*vec[:,0] + mat[2][1]*vec[:,1] + mat[2][2]*vec[:,2]
    return res


def all_inside(pos3d, center, r_search):
    try:
        tree = KDTree(np.squeeze(pos3d))
        in_halo = tree.query_radius(center,r_search)[0]
    except:
        sys.exit("Nope")
    return in_halo

def half_mass(mass,r):
    """
    returns half mass radius
    i.e the radius at which the half of the total 
    mass is contained
    input:
    mass = array of particles mass
    r =  array of particles distance from center 
    """
    m_tot, m_sort = mass.sum() / 2., mass[np.argsort(r)]
    aux,counter = 0,-1
    while aux < m_tot:
        counter += 1
        aux += m_sort[counter]
    return r[counter]

def check_particles(path):
    """
    this routine check the particle number in  the simulation
    by reading the header file in the snapshot file and using
    as input the path to the stapshot.
    """
    list = glob.glob(path+"/header_?????.txt")
    linum = n = 0
    part_arrays = fi = np.array([])
    for f in list:
        fi = np.append(fi,f)
        n+=1
    myfi = open(fi[0])
    for l in myfi:
        linum += 1
        if linum%2==0 and linum<=8:
            row = l.split(' ')
            part_arrays = np.append(part_arrays,np.float64(row[-1]))
    n_tot = part_arrays[0]
    n_dm = part_arrays[1]
    n_st = part_arrays[2]
    if (n_tot>0):
        return n_tot, n_dm, n_st
    else:
        myfi, linum = open(fi[0]), -1
        for l in myfi:
            linum += 1
            row = l.split('  ')
            if linum==7:
                n_dm = float(row[-1])
            if linum==8:
                n_st = float(row[-1])
        n_tot = n_st + n_dm
        return int(n_tot), int(n_dm), int(n_st)





def mass_distribution_tensor(mass,pos,return_vec=False):
        """
        calculates the mass distibrution tensor as in 
        equations 2 of arXiv:1207.4555v2
        """
        P = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
               P[i][j] = np.sum(mass*pos[:,i]*pos[:,j])
        eigen_values,evecs = np.linalg.eig(P)
        order = np.argsort(abs(eigen_values))
        T = np.zeros((3,3))
        T[0],T[1],T[2] = evecs[:,order[2]],evecs[:,order[1]],evecs[:,order[0]]
        D = np.dot(T,np.dot(P,np.transpose(T)))
        if (return_vec):
            return D,T,evecs,eigen_values
        else:
            return D,T



def m_matrix_for_r(halo,comp,r):
        if comp=='stars':
           sim = halo.st
        elif comp=='halo':
           sim = halo.dm
        else:
           sys.exit('Not a valid component')
        sim.r = np.sqrt((sim.pos3d[:,0])**2 +(sim.pos3d[:,1])**2 +(sim.pos3d[:,2])**2 )
        pos_selection = sim.pos3d[(sim.r<r)]
        mass_selection = sim.mass[(sim.r<r)]
        D,T = mass_distribution_tensor(mass_selection,pos_selection)
        return D,T


def read_arguments():
    for i in range(len(sys.argv)):
        if sys.argv[i]=='-h':
            if sys.argv[i+1]=='halo_B':
                h = halo_info.HALOB()
            elif sys.argv[i+1]=='halo_A':
                h = halo_info.HALOA()
            elif sys.argv[i+1]=='mochima':
                h = halo_info.MOCHIMA()
            else:
                sys.exit(">> no halo have been defined")
    return h

#### Ploting rutines ####

def face_on_dm(sim,lims,points):
    edges = np.linspace(lims[0],lims[1],points)
    H, xedges, yedges = np.histogram2d(sim.dm.pos3d[:,0], 
                                       sim.dm.pos3d[:,1],
                                       bins=(edges, edges),
                                       weights=sim.dm.mass)
    result = H.T
    return result, edges

def face_on_st(sim,lims,points,thikness=.5):
    disk = (np.abs(sim.st.pos3d[:,2])<thikness)
    edges = np.linspace(lims[0],lims[1],points)
    H, xedges, yedges = np.histogram2d(sim.st.pos3d[disk,0], 
                                       sim.st.pos3d[disk,1],
                                       bins=(edges, edges),
                                       weights=sim.st.mass[disk])
    result = H.T
    return result, edges

def face_on_gs_temp(sim,lims,points,thikness=.9):
    disk = (np.abs(sim.gs.pos3d[:,2])<thikness)
    edges = np.linspace(lims[0],lims[1],points)
    H, xedges, yedges = np.histogram2d(sim.gs.pos3d[disk,0], 
                                       sim.gs.pos3d[disk,1],
                                       bins=(edges, edges),
                                       weights=sim.gs.temp[disk])
    result = H.T
    return result, edges

def face_on_gs(sim,lims,points,thikness=.9):
    disk = (np.abs(sim.gs.pos3d[:,2])<thikness)
    edges = np.linspace(lims[0],lims[1],points)
    H, xedges, yedges = np.histogram2d(sim.gs.pos3d[disk,0], 
                                       sim.gs.pos3d[disk,1],
                                       bins=(edges, edges),
                                       weights=sim.gs.mass[disk])
    result = H.T
    return result, edges

def edge_on_st(sim,lims,points):
    #disk = sim.st.pos3d[:,2]
    edges = np.linspace(lims[0],lims[1],points)
    H, xedges, yedges = np.histogram2d(sim.st.pos3d[:,0], 
                                       sim.st.pos3d[:,2],
                                       bins=(edges, edges),
                                       weights=sim.st.mass)
    result = H.T
    return result, edges
    
def edge_on_gs_temp(sim,lims,points):
    edges = np.linspace(lims[0],lims[1],points)
    H, xedges, yedges = np.histogram2d(sim.gs.pos3d[:,0], 
                                       sim.gs.pos3d[:,2],
                                       bins=(edges, edges),
                                       weights=sim.gs.temp)
    result = H.T
    return result, edges
def edge_on_gs(sim,lims,points):
    edges = np.linspace(lims[0],lims[1],points)
    H, xedges, yedges = np.histogram2d(sim.gs.pos3d[:,0], 
                                       sim.gs.pos3d[:,2],
                                       bins=(edges, edges),
                                       weights=sim.gs.mass)
    result = H.T
    return result, edges
