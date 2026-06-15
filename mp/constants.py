"""Project-wide constants.

mug_mesh.obj is a closed solid (no modeled cavity). Its OPENING direction is
mesh local -y, established by a side-view render: under the old +y-up convention
the cup rendered upside-down, so -y is the opening. In the mug's *local* frame:
  -y  →  cup opening (the "up" direction when the cup stands upright)
  +y  →  cup base
  +x  →  body lateral
  +z  →  body lateral + handle protrusion

T_ee_to_mug (saved in data/ee_to_mug.npy) uses this raw local frame — we do NOT
re-normalize the mesh; instead, every cost function uses MUG_UP_AXIS_LOCAL
explicitly when computing the cup's world-frame up vector.
"""
import numpy as np

MUG_UP_AXIS_LOCAL = np.array([0.0, -1.0, 0.0])   # mesh -y is the cup opening (up)

# Rotation that places the mug upright in world frame: R_x(-π/2).
# Maps mug local -y (opening) → world +z, so the cup stands opening-up.
R_UPRIGHT = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0],
], dtype=float)
