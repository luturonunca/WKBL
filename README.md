### WKBL ###

WKBL provides utilities to load, center, and rotate RAMSES-style galaxy
snapshots. The main entry point is `Galaxy_Hound` in `wkbl/astro/galaxy_peeker.py`.

#### Load a snapshot

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
