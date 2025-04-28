import numpy as np
from . import  _dark_matter as d
from . import _stars as s
from . import _gas as g
import glob
from . import nbody_essentials as nbe


################################################################################
class Galaxy_Hound:
    def __init__(self, file_path,getcen=False,**kwargs):
        """
        Main Object: creates and object that loads all galaxy components and
        gives axes for numpy arrays for every variable from every particle 
        or cell.
        """
        # save path to snapshot
        self.file = file_path
        halonu = len(glob.glob(file_path+"/halo*"))
        if halonu==0:newage = False
        else:newage = True
        # initialize simu parameters
        self.p = nbe.Info_sniffer(file_path, newage=newage)
        # dicern on ramses versions

        ############################### ARGUMENTS ##############################
        gas             = kwargs.get('gas'             ,True ) # Load gas
        self.rmax_rot   = kwargs.get('rmax_rot'        ,10   ) # rmax for rotation
        self.quiet      = kwargs.get('quiet'           ,False) # avoid printing
        hsml            = kwargs.get('hsml'            ,False) # force a res level
        self.dmo        = kwargs.get('dmo'             ,False) # dark matter only
        self.flush      = kwargs.get('flush'           ,False) # !!!!!!!!! Check
        dens            = kwargs.get('dens'            ,False) # compute densities
        self.p.SHIFT    = kwargs.get('shift'           ,False) # shift inside cells
        comov           = kwargs.get('comov'           ,True ) # comoving coordinates
        rockstar_path   = kwargs.get('rockstar_path'   ,"" )   #
        ########################################################################
        # flags for components
        self._dms, self._sts, self._gss  = False, False, False
        # inizialize mean halo velocity
        halo_vel = kwargs.get('halo_vel',[0.,0.,0.])    ########
        ############################### load data ##############################
        # read headers
        self.n_tot, self.n_dm, self.n_st = nbe.check_particles(file_path)
        # dark matter
        if self.n_dm > 0:
            if not self.quiet: print("loading Dark matter..")
            self.dm = d._dark_matter(file_path,self.p,comov=comov,rs=rockstar_path)
            self._dms = True
        # stars
        if self.n_st > 0 and self.dmo==False:
            if not self.quiet: print("loading Stars..")
            self.st = s._stars(file_path,self.p, comov=comov)
            self._sts = True
            # where there is stars there is gas
            if  gas==True:
                if not self.quiet: print("loading Gas..")
                self.gs = g._gas(file_path, self.p,comov=comov)
                self._gss = True
        else:
            self.dmo = True

    def r_virial(self,r_max=600,r_min=0,rotate=True,n=2.5,bins=512):
        """
        once the center have been defined this function computes
        several virial radii for the galaxy:
        r_200 : the radius where the spherical density is 200 times
        the critical density of the universe
        r_97 : the radius where the spherical density is 97 times
        the critical density of the universe
        r_BN : virial radius as defined by Brian & Norman 1998
        """
        # initializing
        rmax_rot = self.rmax_rot
        positions = np.array([], dtype=np.int64).reshape(0,3)
        masses = np.array([], dtype=np.int64)
        # stack available masses and positions 
        if (self._dms):
            positions = np.vstack([positions,self.dm.pos3d])
            masses = np.append(masses,self.dm.mass)
        if (self._sts):
            positions = np.vstack([positions,self.st.pos3d])
            masses = np.append(masses,self.st.mass)
        if (self._gss):
            positions = np.vstack([positions,self.gs.pos3d])
            masses = np.append(masses,self.gs.mass)
        # find virial radii 
        r = np.sqrt((positions[:,0])**2 +(positions[:,1])**2 +(positions[:,2])**2 )
        a,b,c,d = nbe.get_radii(r,masses,self.p,r_max,bins=bins)
        self.delta_crit, self.r200,self.r97,self.rBN = a,b,c,d
        rnot = False
        if (rotate)and(self._sts):
            if (self.flush):self.redefine(n,simple=True)
            if self.p.Z>2:
                self.rotate_galaxy(rmin=0.5,rmax=rmax_rot)
            else:
                self.rotate_galaxy()
            D = np.dot(self.matrix_T,np.dot(self.matrix_P,np.transpose(self.matrix_T)))
            if not self.quiet: nbe.print_matrix(D)

        if (rotate)and(self.dmo):
            if self.p.Z>2:
                self.rotate_galaxy(rmin=0.5,rmax=rmax_rot)
            else:
                self.rotate_galaxy(rmin=3,rmax=20,comp='dm')
            D = np.dot(self.matrix_T,np.dot(self.matrix_P,np.transpose(self.matrix_T)))
            if not self.quiet: nbe.print_matrix(D)

        self.frame_of_ref(r)
        self.redefine(n)

    def center_shift(self,nucenter):
        self.center = np.zeros(3)
        if (self._dms):
            self.dm.shift(nucenter)
        if (self._sts):
            self.st.shift(nucenter)
        if (self._gss):
            self.gs.shift(nucenter)

    def redefine(self,n,simple=False):
        if (self._dms):
            self.dm.halo_Only(self.center, n, self.rBN,simple=simple)
        if (self._sts):
            self.st.halo_Only(self.center, n, self.rBN,simple=simple)
        if (self._gss):
            self.gs.halo_Only(self.center, n, self.rBN,simple=simple)

    def rotate_galaxy(self,rmin=3,rmax=10,comp='st',affect=True):
        if comp == 'st':
            r2 = (self.st.pos3d[:,0])**2 +(self.st.pos3d[:,1])**2 +(self.st.pos3d[:,2])**2
            pos_ring = self.st.pos3d[(r2<rmax**2)&(r2>rmin**2)]
        elif comp== 'dm':
            r2 = (self.dm.pos3d[:,0])**2 +(self.dm.pos3d[:,1])**2 +(self.dm.pos3d[:,2])**2
            pos_ring = self.dm.pos3d[(r2<rmax**2)&(r2>rmin**2)]

        P = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                first = np.mean(pos_ring[:,i]*pos_ring[:,j])
                second =(np.mean(pos_ring[:,i])*np.mean(pos_ring[:,j]))
                P[i][j] = first - second
        eigen_values,evecs = np.linalg.eig(P)
        order = np.argsort(abs(eigen_values))
        T = np.zeros((3,3))
        T[0],T[1],T[2] = evecs[:,order[2]],evecs[:,order[1]],evecs[:,order[0]]
        self.matrix_T = T
        self.matrix_P = P
        if (self._dms) and (affect):
            self.dm.rotate(T)
        if (self._sts) and (affect):
            self.st.rotate(T)
        if (self._gss) and (affect):
            self.gs.rotate(T)

    def frame_of_ref(self,r):
        ############################################################
        # substract the average speed of the system
        positions = vels = np.array([], dtype=np.int64).reshape(0,3)
        mass = np.array([])
        if (self._dms):
            #r = np.append(r,self.dm.r)
            mass = np.append(mass,self.dm.mass)
            vels = np.vstack([vels,self.dm.vel3d])

        if (self._sts):
            #r = np.append(r,self.st.r)
            mass = np.append(mass,self.st.mass)
            vels = np.vstack([vels,self.st.vel3d])
        if (self._gss):
            #r = np.append(r,self.gs.r)
            mass = np.append(mass,self.gs.mass)
            vels = np.vstack([vels,self.gs.vel3d])
        # velocity of the center of mass 
        sel = np.where(r<self.r200)
        self.com_vx = np.sum(mass[sel]*vels[sel,0])/np.sum(mass[sel])
        self.com_vy = np.sum(mass[sel]*vels[sel,1])/np.sum(mass[sel])
        self.com_vz = np.sum(mass[sel]*vels[sel,2])/np.sum(mass[sel])
        if (self._dms):self.dm.vel_frame(self.com_vx,self.com_vy,self.com_vz)
        if (self._sts):self.st.vel_frame(self.com_vx,self.com_vy,self.com_vz)
        if (self._gss):self.gs.vel_frame(self.com_vx,self.com_vy,self.com_vz)
        ##########################################################


