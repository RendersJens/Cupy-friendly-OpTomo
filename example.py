import numpy as np
import cupy as cp
from cupy_optomo import OpTomo
import tomopy
import astra
from time import time

im_size = 256
x = cp.asarray(tomopy.shepp3d(im_size), dtype=np.float32)

vol_geom = astra.create_vol_geom(x.shape[1], x.shape[2], x.shape[0])
proj_geom = astra.create_proj_geom('parallel3d', 1, 1, x.shape[1], x.shape[2], np.linspace(0, np.pi, 200))
proj_id = astra.create_projector('cuda3d', proj_geom, vol_geom)

# the astra version of OpTomo only accepts numpy arrays (CPU)
# so we have to copy x to CPU before we can pass it to W1.
W1 = astra.OpTomo(proj_id)

# This would crash:
# p1 = W1 @ x.ravel()

# so we do this instead
t0 = time()
p1 = W1 @ (x.ravel().get())
print("Projected with ASTRA OpTomo in", time() - t0, "seconds")

# with the cupy friendly OpTomo we can work on cupy arrays (GPU)
# directly (and it still works on numpy arrays)
W = OpTomo(proj_id)

t0 = time()
p2 = W @ x.ravel()
print("Projected with Cupy friendly OpTomo in", time() - t0, "seconds")

print("all close?", np.allclose(p1, p2.get()))