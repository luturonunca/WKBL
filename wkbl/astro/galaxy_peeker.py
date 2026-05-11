"""
Galaxy_Hound: load, center, and rotate RAMSES-style galaxy snapshots.
"""
import glob
import numpy as np
import yt
from . import _dark_matter as d
from . import component as comp
from . import _stars as s
from . import _gas as g
from . import _bns as b
from . import nbody_essentials as nbe


################################################################################
def _read_header_counts(file_path):
    header = glob.glob(file_path + "/header_?????.txt")
    if not header:
        return {}
    counts = {}
    with open(header[0], "r") as handle:
        for line in handle:
            row = line.strip().split()
            if len(row) != 2:
                continue
            name, count = row
            try:
                counts[name] = int(count)
            except ValueError:
                continue
    return counts


def _header_summary(header_counts, dm_count, st_count, gas_count, bns_count):
    return {
        "dm_particles": int(dm_count),
        "star_particles": int(st_count),
        "gas_cells": int(gas_count),
        "bns_particles": int(bns_count),
        "raw_header": dict(header_counts),
    }


def _ensure_bns_filter(ds):
    def bns_filter(pfilter, data):
        return data[("io", "particle_family")] == 6

    try:
        yt.add_particle_filter(
            "bns",
            function=bns_filter,
            filtered_type="io",
            requires=["particle_family"],
        )
    except Exception:
        pass
    try:
        ds.add_particle_filter("bns")
    except Exception:
        pass


class Galaxy_Hound:
    def __init__(self, file_path,getcen=False,**kwargs):
        """
        Load a simulation snapshot and initialize particle components.

        This is the main entry point for analysis workflows. It loads dark
        matter, stars, and gas (if present), exposes their particle arrays,
        and prepares metadata through Info_sniffer.

        Parameters
        ----------
        file_path : str
            Path to the snapshot directory.
        getcen : bool, optional
            If True, attempt to compute and store the halo center on init.
        gas : bool, optional
            Load gas particles if available.
        dmo : bool, optional
            Dark-matter-only mode (skip stars/gas).
        comov : bool, optional
            Use comoving coordinates if True.
        rockstar_path : str, optional
            Path to Rockstar catalogs for subhalo data.
        quiet : bool, optional
            Suppress console output when True.

        Notes
        -----
        Typical workflow:
        - load with Galaxy_Hound(...)
        - compute center (nbody_essentials.real_center or getcen)
        - shift with center_shift(...)
        - compute virial radii with r_virial(...)
        - rotate with rotate_galaxy(...)
        """
        # save path to snapshot
        self.file = file_path
        halonu = len(glob.glob(file_path+"/halo*"))
        if halonu==0:newage = False
        else:newage = True
        # initialize simu parameters
        self.p = nbe.Info_sniffer(file_path, newage=newage)
        # dicern on ramses versions
        self.ds = yt.load(file_path)
        self.ad = self.ds.all_data()
        self._tcur_code = float(self.ds.current_time.value)

        ############################### ARGUMENTS ##############################
        gas             = kwargs.get('gas'             ,True ) # Load gas
        bns             = kwargs.get('bns'             ,True ) # Load BNS particles
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
        self._dms, self._sts, self._gss, self._bns = False, False, False, False
        # inizialize mean halo velocity
        halo_vel = kwargs.get('halo_vel',[0.,0.,0.])    ########
        ############################### load data ##############################
        # read headers
        self.n_tot, self.n_dm, self.n_st = nbe.check_particles(file_path)
        self.header_counts = _read_header_counts(file_path)
        # dark matter
        if self.n_dm > 0:
            if not self.quiet: print("loading Dark matter..")
            self.dm = d._dark_matter(
                file_path,
                self.p,
                comov=comov,
                rs=rockstar_path,
                ds=self.ds,
                ad=self.ad,
            )
            self._dms = True
            dm_count = len(self.dm.mass)
            self.ad.clear_data()
        else:
            dm_count = 0
        # bns particles
        if bns and self.header_counts.get("bns", 0) > 0:
            if not self.quiet: print("loading BNS..")
            _ensure_bns_filter(self.ds)
            self.bns = b._bns(
                file_path,
                self.p,
                comov=comov,
                ds=self.ds,
                ad=self.ad,
            )
            self._bns = True
            bns_count = len(self.bns.mass)
            self.ad.clear_data()
        else:
            bns_count = 0
        # stars
        if self.n_st > 0 and self.dmo==False:
            if not self.quiet: print("loading Stars..")
            self.st = s._stars(file_path, self.p, comov=comov, ds=self.ds, ad=self.ad)
            self._sts = True
            st_count = len(self.st.mass)
            self.ad.clear_data()
            # where there is stars there is gas
            if  gas==True:
                if not self.quiet: print("loading Gas..")
                self.gs = g._gas(file_path, self.p, comov=comov, ds=self.ds, ad=self.ad)
                self._gss = True
                gas_count = len(self.gs.mass)
            else:
                gas_count = 0
        else:
            self.dmo = True
            st_count = 0
            gas_count = 0

        self.header = _header_summary(
            self.header_counts, dm_count, st_count, gas_count, bns_count
        )
        # Current cosmic time in Gyr, consistent with star/BNS age convention.
        self.current_time = self.p.t_cur_gyr
        # yt was only a loading intermediary — release everything once arrays are built.
        try:
            self.ds.index.clear_all_caches()
        except Exception:
            pass
        for _c in ("dm", "st", "gs", "bns"):
            _obj = getattr(self, _c, None)
            if _obj is not None:
                _obj._ad = None
                _obj._ds = None
        self.ad = None
        self.ds = None

    def r_virial(self,r_max=600,r_min=0.5,rotate=True,n=2.5,bins=512):
        """
        Compute virial radii and optionally rotate to a disk frame.

        Requires particle positions to be centered. This computes:
        - r_200: density = 200 * rho_crit
        - r_97 : density = 97 * rho_crit
        - r_BN : Bryan & Norman (1998) virial radius

        Parameters
        ----------
        r_max : float
            Maximum radius (kpc) for radial sampling.
        r_min : float
            Minimum radius (kpc) used for rotation window.
        rotate : bool
            If True, rotate galaxy to align with its disk.
        n : float
            Halo selection multiplier (n * r200) for redefine().
        bins : int
            Number of radial bins used in the density profile.
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
        if (self._bns):
            positions = np.vstack([positions,self.bns.pos3d])
            masses = np.append(masses,self.bns.mass)
        # find virial radii 
        r = np.sqrt((positions[:,0])**2 +(positions[:,1])**2 +(positions[:,2])**2 )
        a,b,c,d = nbe.get_radii(r,masses,self.p,r_max,bins=bins)
        self.delta_crit, self.r200,self.r97,self.rBN = a,b,c,d
        rnot = False
        if (rotate)and(self._sts):
            if (self.flush):self.redefine(n,simple=True)
            if self.p.Z>2:
                self.rotate_galaxy(rmin=r_min,rmax=rmax_rot)
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
        """
        Shift all loaded components to a new center.

        Parameters
        ----------
        nucenter : array-like, shape (3,)
            New center in the current coordinate system.
        """
        self.center = np.zeros(3)
        if (self._dms):
            self.dm.shift(nucenter)
        if (self._sts):
            self.st.shift(nucenter)
        if (self._gss):
            self.gs.shift(nucenter)
        if (self._bns):
            self.bns.shift(nucenter)

    def redefine(self,n,simple=False):
        """
        Restrict loaded components to the halo region (n * rBN).

        Parameters
        ----------
        n : float
            Multiplier for the virial radius defining the halo region.
        simple : bool, optional
            If True, skip expensive fields where supported.
        """
        if (self._dms):
            self.dm.halo_Only(self.center, n, self.rBN,simple=simple)
        if (self._sts):
            self.st.halo_Only(self.center, n, self.rBN,simple=simple)
        if (self._gss):
            self.gs.halo_Only(self.center, n, self.rBN,simple=simple)
        if (self._bns):
            self.bns.halo_Only(self.center, n, self.rBN,simple=simple)

    def rotate_galaxy(self,rmin=3,rmax=10,comp='st',affect=True):
        """
        Align the galaxy to a disk frame using the mass distribution tensor.

        Parameters
        ----------
        rmin, rmax : float
            Radial window (kpc) used to define the disk plane.
        comp : str
            Component to use for the rotation ("st" or "dm").
        affect : bool
            If True, apply the rotation to loaded components.
        """
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
        if (self._bns) and (affect):
            self.bns.rotate(T)

    def frame_of_ref(self,r):
        """
        Compute halo velocity frame and shift velocities accordingly.

        Parameters
        ----------
        r : array-like
            Radial distances used to select particles for the frame.
        """
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
        if (self._bns):
            mass = np.append(mass,self.bns.mass)
            vels = np.vstack([vels,self.bns.vel3d])
        # velocity of the center of mass 
        sel = np.where(r<self.r200)
        self.com_vx = np.sum(mass[sel]*vels[sel,0])/np.sum(mass[sel])
        self.com_vy = np.sum(mass[sel]*vels[sel,1])/np.sum(mass[sel])
        self.com_vz = np.sum(mass[sel]*vels[sel,2])/np.sum(mass[sel])
        if (self._dms):self.dm.vel_frame(self.com_vx,self.com_vy,self.com_vz)
        if (self._sts):self.st.vel_frame(self.com_vx,self.com_vy,self.com_vz)
        if (self._gss):self.gs.vel_frame(self.com_vx,self.com_vy,self.com_vz)
        if (self._bns):self.bns.vel_frame(self.com_vx,self.com_vy,self.com_vz)
        ##########################################################

    def print_header(self):
        """
        Print a short summary of particle and cell counts.
        """
        hdr = self.header
        print("Header summary")
        print("  DM particles  :", hdr.get("dm_particles", 0))
        print("  stars         :", hdr.get("star_particles", 0))
        print("  gas cells     :", hdr.get("gas_cells", 0))
        print("  bns           :", hdr.get("bns_particles", 0))
