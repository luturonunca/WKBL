### WKBL ###

WKBL provides utilities to load, center, and rotate RAMSES-style galaxy
snapshots. The main entry point is `Galaxy_Hound` in `wkbl/astro/galaxy_peeker.py`.

#### Load a snapshot

This uses `yt` to read RAMSES outputs.

```python
from wkbl.astro.galaxy_peeker import Galaxy_Hound

halo = Galaxy_Hound("/path/to/snapshot", comov=True, quiet=False)
```

#### Center the galaxy

Use the shrinking-sphere method to find the center from stars (or another
component), then shift all components:

```python
from wkbl.astro import nbody_essentials as nbe

center = nbe.real_center(halo.st.pos3d, halo.st.mass, n=7000)
halo.center_shift(center)
```

#### Compute virial radii and rotate to disk frame

```python
halo.r_virial(r_max=600, r_min=0.5, rotate=True, n=2.5, bins=512)
```

This computes `r200`, `r97`, and `rBN` and (optionally) aligns the galaxy to
its disk plane using the mass distribution tensor.

#### Typical analysis helper

Below is a common helper used at the start of analysis scripts. It loads a
snapshot, recenters, computes virial quantities, and estimates densities.

```python
import numpy as np
import wkbl
from wkbl.astro import nbody_essentials as nbe
import unsiotools.simulations.cfalcon as falcon

cf = falcon.CFalcon()

def load_halo(path, rs_path, rotate=True, DM=False, gas=False, n=1, virial=True):
    # load simulation
    myhalo = wkbl.Galaxy_Hound(path, rockstar_path=rs_path, gas=gas)
    # define zoom region
    zoomreg = np.where(myhalo.dm.mass == myhalo.dm.mass.min())
    # calculate center with zoom DM particles or stars
    if DM:
        print("center in DM")
        center = nbe.real_center(myhalo.dm.pos3d[zoomreg], myhalo.dm.mass[zoomreg])
    else:
        center = nbe.real_center(myhalo.st.pos3d, myhalo.st.mass)
    # recenter the whole thing
    myhalo.center_shift(center)
    # compute virial radii and cut data with r > n * rvir
    if not virial:
        return myhalo
    myhalo.r_virial(r_max=600, n=20, rotate=rotate)
    pos = np.array(myhalo.dm.pos3d.reshape(len(myhalo.dm.pos3d) * 3),
                   dtype=np.float32)
    mass = np.array(myhalo.dm.mass, dtype=np.float32)
    ok, myhalo.dm.rho, _ = cf.getDensity(pos, mass)
    try:
        inside = np.where(myhalo.st.r < 3 * 0.14)[0]
        ranarray = 0.035 * 0.5 * (2 * np.random.rand(len(inside), 3) - 0.5)
        myhalo.st.pos3d[inside] = myhalo.st.pos3d[inside] + ranarray
        pos = np.array(myhalo.st.pos3d.reshape(len(myhalo.st.pos3d) * 3),
                       dtype=np.float32)
        mass = np.array(myhalo.st.mass, dtype=np.float32)
        ok, myhalo.st.rho, _ = cf.getDensity(pos, mass, ncrit=15)
    except Exception:
        print("no density for stars.. maybe no stars")
    return myhalo

mygalaxy = load_halo(path2, rs_path, DM=DM, rotate=rotate, gas=True)
```
