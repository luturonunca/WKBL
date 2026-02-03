import glob
import warnings

import numpy as np
from matplotlib import rc
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from numba import njit, prange

warnings.filterwarnings("ignore")
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
## for Palatino and other serif fonts use:
# rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

def get_zfrominfo(simpath,num):
    path = simpath+"/output_{0:05d}".format(num)+"/info_{0:05d}".format(num)+".txt"
    file = open(path)
    for l in file:
        if l[:4]!="aexp":continue
        aexp = np.float(l.split("=")[-1])
        z    = 1/aexp -1 
        

    file.close()
    return z,path[:-15]

def get_info(num,path):
    _vars=dict()
    #print(path+"/info*".format(num))
    file = glob.glob(f"{path}/info*")[0]

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
    return _vars



@njit(parallel=True)
def project_cells_split_rotate(x, y, z, cell_size, quantity, quantity_type,
                               img_shape, bounds, pixel_size, max_subcell, R):
    """
    Split parent cells into subcells of side <= max_subcell, rotate each subcell center
    by rotation matrix R (3x3), project onto rotated x-y plane and accumulate into image.

    Parameters
    ----------
    x,y,z : parent cell centers
    cell_size : parent cell side
    quantity : per parent cell quantity
    quantity_type : 0 extensive / 1 intensive
    img_shape : (nx, ny)
    bounds : (xmin, xmax, ymin, ymax) in rotated-frame coordinates (i.e. target plane coords)
    pixel_size : pixel_size in same units as coords
    max_subcell : max allowed subcell side
    R : 3x3 rotation matrix (applies to parent->rotated frame); use numpy array passed in

    Returns
    -------
    img (nx,ny) accumulated, for intensive returns averaged (like your routine)
    """
    nx, ny = img_shape
    img = np.zeros((nx, ny), dtype=np.float64)
    weight = np.zeros((nx, ny), dtype=np.float64)

    xmin, xmax, ymin, ymax = bounds

    N = x.shape[0]
    for i in prange(N):
        cx = x[i]
        cy = y[i]
        cz = z[i]
        l = cell_size[i]
        q = quantity[i]

        # subdivisions per axis
        n = int(np.ceil(l / max_subcell))
        if n < 1:
            n = 1
        ls = l / n  # subcell side
        x0 = cx - 0.5 * l
        y0 = cy - 0.5 * l
        z0 = cz - 0.5 * l

        # per-subcell quantity handling
        if quantity_type == 0:
            q_sub = q / (n * n * n)
        else:
            q_sub = q

        for ix in range(n):
            sx = x0 + (ix + 0.5) * ls
            for iy in range(n):
                sy = y0 + (iy + 0.5) * ls
                for iz in range(n):
                    sz = z0 + (iz + 0.5) * ls

                    # Rotate the subcell center: r = R @ [sx, sy, sz]
                    rx = R[0,0]*sx + R[0,1]*sy + R[0,2]*sz
                    ry = R[1,0]*sx + R[1,1]*sy + R[1,2]*sz
                    # rz = R[2,0]*sx + R[2,1]*sy + R[2,2]*sz  # not used for projection

                    # Now project onto rotated-frame XY plane using rx, ry
                    x0p = rx - 0.5 * ls
                    x1p = rx + 0.5 * ls
                    y0p = ry - 0.5 * ls
                    y1p = ry + 0.5 * ls

                    ix0 = max(0, int((x0p - xmin) / pixel_size))
                    ix1 = min(nx, int((x1p - xmin) / pixel_size) + 1)
                    iy0 = max(0, int((y0p - ymin) / pixel_size))
                    iy1 = min(ny, int((y1p - ymin) / pixel_size) + 1)

                    for iix in range(ix0, ix1):
                        px0 = xmin + iix * pixel_size
                        px1 = px0 + pixel_size
                        ox = max(0.0, min(x1p, px1) - max(x0p, px0))
                        if ox <= 0.0:
                            continue
                        for iiy in range(iy0, iy1):
                            py0 = ymin + iiy * pixel_size
                            py1 = py0 + pixel_size
                            oy = max(0.0, min(y1p, py1) - max(y0p, py0))
                            if oy <= 0.0:
                                continue
                            area_overlap = ox * oy
                            if area_overlap > 0.0:
                                if quantity_type == 0:
                                    # q_sub is subcell total (extensive); distribute by area fraction
                                    img[iix, iiy] += q_sub * (area_overlap / (ls * ls))
                                else:
                                    # q_sub is intensive; accumulate q*area, and accumulate area as weight
                                    img[iix, iiy] += q_sub * area_overlap
                                    weight[iix, iiy] += area_overlap

    if quantity_type == 1:
        # finalize averaging
        for iix in range(nx):
            for iiy in range(ny):
                if weight[iix, iiy] > 0.0:
                    img[iix, iiy] /= weight[iix, iiy]
                else:
                    img[iix, iiy] = 0.0

    return img


def build_rotation_from_angmom(pos, vel, mass=None, prev_R=None, eps=1e-12):
    """
    Build rotation matrix R (3x3) that maps world coords -> frame where angular momentum is +z.

    Parameters
    ----------
    pos : (N,3) numpy array
        positions of tracer particles (e.g. pos_ring)
    vel : (N,3) numpy array
        velocities of tracer particles (same order)
    mass : (N,) or None
        optional weights per particle; if None equal weights are used
    prev_R : (3,3) or None
        optional previous rotation matrix used to enforce temporal sign-continuity
    eps : small float to avoid divide-by-zero

    Returns
    -------
    R : (3,3) float64
        orthonormal rotation matrix such that R @ L_unit == [0,0,1] (within numerical noise)
    Lvec : (3,) float64
        computed angular momentum vector (weighted)
    """

    # compute (weighted) angular momentum vector L = sum_i m_i (r_i x v_i)
    if mass is None:
        w = np.ones(pos.shape[0], dtype=np.float64)
    else:
        w = mass.astype(np.float64)

    # r x v per particle
    cross = np.cross(pos, vel)  # shape (N,3)
    # weighted sum
    Lvec = (w[:,None] * cross).sum(axis=0)
    Lnorm = np.linalg.norm(Lvec)
    if Lnorm < eps:
        raise ValueError("Angular momentum vector is essentially zero.")

    w_z = Lvec / Lnorm  # target +z axis (world coords)

    # choose a stable reference vector to build x-axis: try world x, if nearly parallel use world y
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(ref, w_z)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # new x-axis = normalize(cross(ref, w_z))
    u = np.cross(ref, w_z)
    u_norm = np.linalg.norm(u)
    if u_norm < eps:
        # fallback (shouldn't usually happen)
        u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        u_norm = np.linalg.norm(u)
    u /= u_norm

    # new y-axis = cross(w_z, u)
    v = np.cross(w_z, u)
    # ensure orthonormal (small numerical error)
    # assemble E with columns [u, v, w_z]
    E = np.column_stack((u, v, w_z))  # shape (3,3)

    # Orthonormalize via SVD to remove tiny numerical errors and ensure det = +/-1
    U, s, Vt = np.linalg.svd(E)
    E_ortho = U @ Vt

    # ensure right-handed with det +1
    if np.linalg.det(E_ortho) < 0:
        # flip sign of the first column (or last) to fix parity
        E_ortho[:, 0] *= -1.0

    # Rotation that maps world -> new frame coordinates:
    R = E_ortho.T  # so coordinates in new frame are r = R @ v

    # Make sure angular momentum maps to positive z in rotated frame:
    # rL = R @ Lvec ; we want rL ≈ [0, 0, +|L|]
    rL = R @ Lvec
    if rL[2] < 0:
        # flip the z-axis (equivalent to flipping third column of E_ortho and row of R)
        E_ortho[:, 2] *= -1.0
        R = E_ortho.T
        rL = R @ Lvec

    # Optional: enforce temporal continuity if prev_R given
    if prev_R is not None:
        # prev_R is rotation mapping world->previous_frame
        # We want columns of E_ortho to align (in sign) with previous basis columns
        prev_E = prev_R.T  # previous E matrix with columns = prev basis in world coords
        E_new = E_ortho.copy()
        for col in range(3):
            dot = np.dot(E_new[:, col], prev_E[:, col])
            if dot < 0:
                # flip sign to keep same orientation as previous grid
                E_new[:, col] *= -1.0
        # re-orthonormalize in case flips introduced small errors
        U, s, Vt = np.linalg.svd(E_new)
        E_new = U @ Vt
        if np.linalg.det(E_new) < 0:
            E_new[:, 0] *= -1.0
        R = E_new.T
        E_ortho = E_new  # update

        # ensure still maps L upward
        rL = R @ Lvec
        if rL[2] < 0:
            E_ortho[:, 2] *= -1.0
            R = E_ortho.T

    return R, Lvec


def rotate_points(R, pos, center=None):
    """
    Apply the same rotation as the AMR projector to point positions.

    Parameters
    ----------
    R : (3,3) ndarray
        Rotation matrix used in the AMR projection (parent -> rotated frame).
    pos : (N,3) ndarray
        Positions in the parent frame (same frame as x,y,z passed to projector).
    center : (3,) array-like or None
        Optional center to subtract before rotation (e.g., galaxy center).
        Use the same center you used (implicitly or explicitly) for the AMR data.

    Returns
    -------
    pos_rot : (N,3) ndarray
        Rotated positions in the rotated frame. pos_rot[:,0] and pos_rot[:,1]
        are the x/y you should plot against the AMR image extent/bounds.
    """
    pos = np.asarray(pos, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    if center is not None:
        c = np.asarray(center, dtype=np.float64)
        pos = pos - c  # IMPORTANT: use the same origin as the AMR grid
    # Match the projector's rx = R[0,:]·[x,y,z]:
    pos_rot = pos @ R.T
    return pos_rot

def rotate_vectors(R, vec):
    """
    Rotate vectors (e.g. velocities) consistently with the AMR rotation.
    No translation applied (vectors don't have origins).
    """
    vec = np.asarray(vec, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    return vec @ R.T



def frame_size_linear(z, size_min, size_max, size_max2,
                      z_min=0.0, z_plateau_lo=4.0, z_plateau_hi=7.0, z_high=100.0):
    """
    Piecewise frame size vs redshift with a size_min plateau:
      - z < z_min:           constant = size_max
      - z in [z_min, 4):     increases linearly size_min -> size_max as z goes 4 -> 0
      - z in [4, 7]:         plateau at size_min
      - z in (7, 100]:       decreases linearly size_min -> size_max2 as z goes 7 -> 100
      - z > 100:             constant = size_max2

    Parameters
    ----------
    z : float or array-like
        Redshift(s).
    size_min : float
        Size at the plateau (and at z=4 and z=7).
    size_max : float
        Size at z=0.
    size_max2 : float
        Size at very high redshift (z >= z_high), e.g., z=100.
    z_min, z_plateau_lo, z_plateau_hi, z_high : floats
        Redshift breakpoints (defaults: 0, 4, 7, 100).
    """
    z = np.asarray(z, dtype=float)
    result = np.empty_like(z, dtype=float)

    # Region A: z > z_high → constant at size_max2
    mask_A = z > z_high
    result[mask_A] = size_max2

    # Region B: (z_plateau_hi, z_high] → linear from size_min (at 7) to size_max2 (at 100)
    mask_B = (z > z_plateau_hi) & (z <= z_high)
    if z_high != z_plateau_hi:
        result[mask_B] = size_min + (size_max2 - size_min) * (
            (z[mask_B] - z_plateau_hi) / (z_high - z_plateau_hi)
        )
    else:
        result[mask_B] = size_min  # degenerate but safe

    # Region C: [z_plateau_lo, z_plateau_hi] → plateau at size_min
    mask_C = (z >= z_plateau_lo) & (z <= z_plateau_hi)
    result[mask_C] = size_min

    # Region D: [z_min, z_plateau_lo) → linear from size_min (at 4) to size_max (at 0)
    mask_D = (z >= z_min) & (z < z_plateau_lo)
    if z_plateau_lo != z_min:
        result[mask_D] = size_min + (size_max - size_min) * (
            (z_plateau_lo - z[mask_D]) / (z_plateau_lo - z_min)
        )
    else:
        result[mask_D] = size_max  # degenerate but safe

    # Region E: z < z_min → constant at size_max
    mask_E = z < z_min
    result[mask_E] = size_max

    # Return scalar if input was scalar
    return result.item() if np.ndim(z) == 0 else result


def gasimagesarrays(simu, rotate=False, rmax=None, rmin=None, outr=None,
                    Xi=0, Yi=1, Zi=2, RI=None, Rx=None,
                    pixel_size=None, max_subcell=None, interp_mode="native",
                    refine_factor=1.0):
    """
    Project gas mass into face-on and edge-on images with AMR-aware subcell splitting.

    Parameters
    ----------
    simu : object
        Simulation container with gas (simu.gs.*) and stars (simu.st.*) arrays.
    rotate : bool
        If True, compute a rotation matrix from the stellar ring (rmin<r<rmax).
    rmax, rmin : float
        Radial bounds (same units as positions) for the stellar ring used in rotation.
    outr : float
        Half-size of the cubic selection box for gas cells (same units as positions).
    Xi, Yi, Zi : int
        Axis indices for projection.
    RI : (3,3) array
        Identity rotation (or base orientation) used when rotate=False.
    Rx : (3,3) array
        Rotation matrix for edge-on view (multiplied as Rx @ T).
    pixel_size : float or None
        Output image pixel size. If None, defaults to min gas cell size in native/refine modes.
    max_subcell : float or None
        Maximum subcell size used for AMR splitting. Controls unigrid-like resolution.
    interp_mode : {"native", "unigrid", "refine"}
        - "native": pixel_size=max_subcell=min(cell_size)
        - "unigrid": requires pixel_size and max_subcell explicitly
        - "refine": uses max_subcell=refine_factor*min(cell_size)
    refine_factor : float
        Subcell size factor used when interp_mode="refine".

    Returns
    -------
    imgface : 2D ndarray
        Face-on projected gas image.
    imgedge : 2D ndarray
        Edge-on projected gas image.
    T : (3,3) ndarray
        Rotation matrix used for the projection.
    """
    if rmax is None or rmin is None:
        raise ValueError("rmax and rmin are required.")
    if outr is None:
        raise ValueError("outr is required.")
    if RI is None:
        raise ValueError("RI is required.")
    if Rx is None:
        raise ValueError("Rx is required.")
    if rotate:
        r2 = (simu.st.pos3d[:, 0])**2 + (simu.st.pos3d[:, 1])**2 + (simu.st.pos3d[:, 2])**2
        pos_ring = simu.st.pos3d[(r2 < rmax**2) & (r2 > rmin**2)]
        vel_ring = simu.st.vel3d[(r2 < rmax**2) & (r2 > rmin**2)]

        ###############################################################################
        ##
        ##                         Rotation Matrix
        ##
        ###############################################################################
        if pos_ring.size == 0 or not np.isfinite(pos_ring).all():
            T = RI
        else:
            P = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    first = np.mean(pos_ring[:, i] * pos_ring[:, j])
                    second = (np.mean(pos_ring[:, i]) * np.mean(pos_ring[:, j]))
                    P[i][j] = first - second
            if not np.isfinite(P).all():
                T = RI
            else:
                eigen_values, evecs = np.linalg.eig(P)
                order = np.argsort(abs(eigen_values))
                T = np.zeros((3, 3))
                T[0], T[1], T[2] = evecs[:, order[2]], evecs[:, order[1]], evecs[:, order[0]]

                E = np.column_stack([evecs[:, order[2]], evecs[:, order[1]], evecs[:, order[0]]])
                T = E.T
    else:
        T = RI

    sel = np.where((np.abs(simu.gs.pos3d[:, 0]) < outr) &
                   (np.abs(simu.gs.pos3d[:, 1]) < outr) &
                   (np.abs(simu.gs.pos3d[:, 2]) < outr))[0]

    # Synthetic data
    N = len(sel)

    x = simu.gs.pos3d[:, Xi][sel]
    y = simu.gs.pos3d[:, Yi][sel]
    z = simu.gs.pos3d[:, Zi][sel]
    cell_size = simu.gs.hsml[sel]
    quantity = simu.gs.mass[sel]

    if interp_mode == "native":
        if pixel_size is None:
            pixel_size = np.min(cell_size)  # match finest resolution
        if max_subcell is None:
            max_subcell = pixel_size
    elif interp_mode == "unigrid":
        if pixel_size is None or max_subcell is None:
            raise ValueError("unigrid mode requires pixel_size and max_subcell.")
    elif interp_mode == "refine":
        if refine_factor <= 0:
            raise ValueError("refine_factor must be > 0.")
        if pixel_size is None:
            pixel_size = np.min(cell_size)
        if max_subcell is None:
            max_subcell = refine_factor * np.min(cell_size)
    else:
        raise ValueError("interp_mode must be 'native', 'unigrid', or 'refine'.")
    print("{0:.3f} pc".format(pixel_size * 1000))
    bounds = (-outr, outr, -outr, outr)   # x/y extent of domain
    nx = int((bounds[1] - bounds[0]) / pixel_size)
    ny = int((bounds[3] - bounds[2]) / pixel_size)
    img_shape = (nx, ny)
    # pixel_size =  2*simu.gs.hsml.min()

    imgface = project_cells_split_rotate(x, y, z, cell_size, quantity, 0,
                                         img_shape, bounds, pixel_size, max_subcell, T)

    imgedge = project_cells_split_rotate(x, y, z, cell_size, quantity, 0,
                                         img_shape, bounds, pixel_size, max_subcell, Rx @ T)
    return imgface, imgedge, T


def maketheplot(ax, img, imgback, Back, gal, R, sinks=True, size=300, app=20,
                square=False, AUTO=True, a=1e2, b=1e4, ffac=1, coords=[0, 1, 2],
                rbondi=1, bounds=None, star_pos=None, star_vel=None, star_sel=None,
                star_r=None, vfac=None, golden_ratio=None):
    if bounds is None:
        bounds = globals().get("bounds")
        if bounds is None:
            raise ValueError("bounds is required for plotting.")
    if golden_ratio is None:
        golden_ratio = globals().get("Goldenratio")
        if golden_ratio is None:
            raise ValueError("golden_ratio (or global Goldenratio) is required.")
    if sinks:
        if star_pos is None:
            star_pos = globals().get("Spos3d")
        if star_vel is None:
            star_vel = globals().get("Svel3d")
        if star_sel is None:
            star_sel = globals().get("Ssel")
        if star_r is None:
            star_r = globals().get("Sr")
        if vfac is None:
            vfac = globals().get("vfac")
        if any(v is None for v in (star_pos, star_vel, star_sel, star_r, vfac)):
            raise ValueError("sinks=True requires star_pos, star_vel, star_sel, star_r, and vfac.")
    if Back:
        if AUTO:a,b=5e3,3e4
        #mass_2 = ax.imshow(masked_data,interpolation='nearest', origin='lower',cmap="Reds",
        #                           extent=[bounds[0], bounds[1], bounds[2], bounds[3]],
        #                          norm=LogNorm(vmin=a,vmax=b),alpha=0.9
        #                      )

        nx, ny = img.shape
        xmin, xmax, ymin, ymax = bounds

        # coordinates of pixel centers
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)

        X, Y = np.meshgrid(x, y, indexing='xy')  # X,Y shape == (ny, nx)
        Z = imgback#.T                     
        ax.contour(X, Y, Z, levels=[7e3, 5e4], colors='r', linewidths=0.5, alpha=1, zorder=10)

    if gal:
        if AUTO:a,b=2e4,8e6
        ax.imshow(img, interpolation='nearest', origin='lower', cmap="Greys",
                  extent=[bounds[0], bounds[1], bounds[2], bounds[3]],
                  norm=LogNorm(vmin=a, vmax=b), alpha=0.8)

    Xi, Yi = coords[0], coords[1]
    #nupos = nbe.matrix_vs_vector(R,Spos3d[Ssel])
    nupos = rotate_points(R, star_pos[star_sel], center=None)
    #nuvel = nbe.matrix_vs_vector(R,Svel3d[Ssel])
    nuvel = rotate_vectors(R, star_vel[star_sel])
    if sinks:
        for i in range(len(star_sel)):
            px,py = ffac*nupos[:,Xi][i],ffac*nupos[:,Yi][i]
            vx,vy =  nuvel[:,Xi][i]/vfac,nuvel[:,Yi][i]/vfac

            if star_r[star_sel][i] == star_r[star_sel].min():
                c="r"

            else:
                c="dodgerblue"
                ax.arrow( px, py, vx, vy, fc=c, ec=c,head_width=0.8, head_length=1 )


            ax.scatter(px,py,marker="x",s=size,c="w",zorder=10)
            ax.scatter(px,py,marker="o",s=size/10,c=c,zorder=10)


            print(i, star_r[i], c, px, py)    
            if rbondi>0:
                ax.add_artist(Circle(xy=(px,py),radius=rbondi,color="r",ls='-',lw=1.2,
                                 fill=False))
            









    #ax.set_ylim([-app*Goldenratio,app*Goldenratio])
    #ax.set_xlim([-app,app])


    height=app
    width = app / golden_ratio
    if square:
        ax.set_ylim([-width,width])
    else:
        ax.set_ylim([-height,height])

    ax.set_xlim([-width,width])
    return width, height
