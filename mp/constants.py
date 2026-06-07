"""Project-wide constants.

mug_mesh.obj has Y-up convention (verified by rendering 4 rotations and observing
that R_x(+90°) makes the cup stand upright). So in the mug's *local* frame:
  +y  →  cup-height (opening direction)
  +x  →  body lateral
  +z  →  body lateral + handle protrusion

T_ee_to_mug (saved in data/ee_to_mug.npy) uses this raw local frame — we do NOT
re-normalize the mesh; instead, every cost function uses MUG_UP_AXIS_LOCAL
explicitly when computing the cup's world-frame up vector.
"""
import numpy as np

MUG_UP_AXIS_LOCAL = np.array([0.0, 1.0, 0.0])    # +y is "up" in the mesh local frame

# Rotation that places the mug upright in world frame: R_x(+π/2).
# Applied as the rotation component of T_world_mug when we want mug local +y → world +z.
R_UPRIGHT = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0],
], dtype=float)
