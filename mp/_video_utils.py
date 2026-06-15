"""Shared helpers for rendering trajectory videos with the CORRECT mug
orientation (opening up).

Convention:
  - Mesh -y is the cup opening direction (verified by raycast).
  - To render any mug pose visually upright, apply a 'mesh flip'
    R_x_mesh_180 on the right side of the world rotation:
        T_render = T_mug_world @ R_x_mesh_180
  - For particle physics, define `MUG_UP_LOCAL = (0, -1, 0)` which, after
    R_render application, points in world +z when the cup is upright.
  - Particles representing water are sampled with mesh-frame y > 0 (the
    'bottom' end of the mug interior cavity). After visual flip, they render
    at world z below the cup center — i.e., the bottom of the visually-upright
    cup. Correct.

Use these utilities in new video scripts to avoid the original upside-down
mug convention used by older `make_*_gif.py` scripts.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import sapien
import PIL.Image
from PIL import ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]

# When the trajectory was planned with the NEW R_UPRIGHT (RX(-π/2)), no extra
# render-time flip is needed (cup is already upright). We keep this identity
# matrix in place to preserve the function signatures of older scripts that
# called `flip_pose_for_render` — those calls now become no-ops.
R_X_MESH_180 = np.eye(3)

# Cup opening direction in mesh local frame.
MUG_UP_LOCAL = np.array([0.0, -1.0, 0.0])


def flip_pose_for_render(T_world_mug: np.ndarray) -> np.ndarray:
    """Apply the visual flip to a 4x4 mug pose so the rendered cup is upright."""
    T = T_world_mug.copy()
    T[:3, :3] = T[:3, :3] @ R_X_MESH_180
    return T


def mat_to_sapien_pose(T: np.ndarray) -> sapien.Pose:
    qxyzw = R.from_matrix(T[:3, :3]).as_quat()
    return sapien.Pose(T[:3, 3].tolist(),
                        [qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]])


def particle_world_position(T_world_mug: np.ndarray, p_local: np.ndarray) -> np.ndarray:
    """Particle world position consistent with the visual flip rendering.
    p_local should be in the *render-correct* mesh frame (water at y > 0)."""
    T_render = flip_pose_for_render(T_world_mug)
    return T_render[:3, :3] @ p_local + T_render[:3, 3]


def sample_water_particles(n: int,
                            radius: float = 0.030,
                            y_low: float = 0.00,
                            y_high: float = 0.045,
                            rng_seed: int = 0) -> np.ndarray:
    """Sample particles in the cup interior (water level near bottom).
    Returns (n, 3) in MESH LOCAL FRAME.
    Mesh +y is the bottom; mesh -y is the opening. Water sits at higher y."""
    rng = np.random.default_rng(rng_seed)
    pts = []
    while len(pts) < n:
        x = rng.uniform(-radius, radius); z = rng.uniform(-radius, radius)
        if x * x + z * z <= radius ** 2:
            pts.append([x, rng.uniform(y_low, y_high), z])
    return np.asarray(pts)


def compute_spill_state(T_world_mug: np.ndarray,
                         a_world: np.ndarray,
                         theta_max_deg: float,
                         g: float = 9.81) -> tuple[bool, float, np.ndarray, np.ndarray]:
    """Returns (spill_state, tilt_deg, g_eff_world, up_local_eff_in_mesh)."""
    R_mug = T_world_mug[:3, :3]
    R_render = R_mug @ R_X_MESH_180
    mug_up_w = R_render @ MUG_UP_LOCAL
    mug_up_w /= np.linalg.norm(mug_up_w) + 1e-9
    G_W = np.array([0.0, 0.0, -g])
    g_eff = G_W - a_world
    g_eff_hat = g_eff / (np.linalg.norm(g_eff) + 1e-9)
    align = -np.dot(g_eff_hat, mug_up_w)
    tilt_deg = np.degrees(np.arccos(np.clip(align, -1.0, 1.0)))
    # In the rendered (flipped) mesh frame, "up" is R_X_MESH_180^T @ MUG_UP_LOCAL
    # = (0, +1, 0). So particles with high +y projection on up_local_eff are
    # near the rendered TOP (closer to opening).
    g_eff_local_render = (R_render).T @ g_eff
    up_local_eff = -g_eff_local_render / (np.linalg.norm(g_eff_local_render) + 1e-9)
    return tilt_deg > theta_max_deg, tilt_deg, g_eff, up_local_eff


def render_camera(scene, w=720, h=540, fovy=0.85,
                    eye=(1.05, -0.75, 0.85), target=(0.5, 0.05, 0.30)):
    from sim.scene import look_at_quat
    cam = scene.add_camera("cam", w, h, fovy=fovy, near=0.05, far=10.0)
    cam.set_pose(sapien.Pose(p=list(eye),
                              q=look_at_quat(list(eye), list(target))))
    return cam


def take_rgb(scene, cam) -> np.ndarray:
    scene.update_render(); cam.take_picture()
    img = cam.get_picture("Color")[..., :3]
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def annotate_frame(img: np.ndarray, title: str, lines: list[tuple[str, str]],
                    title_bg=(220, 235, 245), pad_bottom=False) -> np.ndarray:
    """Add a title bar (top) and a bottom-right info panel.
    lines: list of (text, color_name) where color_name in {'g','r','b','k'}."""
    im = PIL.Image.fromarray(img.copy()); draw = ImageDraw.Draw(im)
    try:
        ft = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        fd = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        ft = ImageFont.load_default(); fd = ImageFont.load_default()
    draw.rectangle([(0, 0), (im.width, 38)], fill=title_bg)
    draw.text((12, 8), title, fill=(20, 20, 20), font=ft)
    if lines:
        box_h = 8 + 22 * len(lines)
        box_w = 240
        draw.rectangle([(im.width - box_w, im.height - box_h),
                         (im.width, im.height)], fill=(245, 245, 245, 220))
        c2rgb = {"g": (30, 110, 30), "r": (170, 30, 30),
                  "b": (40, 60, 170), "k": (20, 20, 20)}
        for i, (txt, col) in enumerate(lines):
            draw.text((im.width - box_w + 8,
                        im.height - box_h + 5 + 22 * i),
                       txt, fill=c2rgb.get(col, (20, 20, 20)), font=fd)
    return np.array(im)


def save_gif(frames: list[np.ndarray], path: Path, duration_ms: int = 100):
    pil = [PIL.Image.fromarray(f) for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=duration_ms, loop=0, optimize=False)
    print(f"saved {path}  ({len(pil)} frames)")


def make_floor_cost(fk, T_em, z_floor=0.05, margin=0.04):
    """Returns a cost fn that penalizes EE / mug going below z_floor + margin.
    Used as `obs_cost_fn` in CHOMPConfig (one term among others)."""
    import torch
    T_em_t = torch.as_tensor(T_em, dtype=fk.dtype, device=fk.device)
    z_thr = z_floor + margin
    def cost(q_traj: torch.Tensor) -> torch.Tensor:
        T = fk(q_traj)                          # (N, 4, 4)
        R = T[:, :3, :3]; t = T[:, :3, 3]
        ee_z = t[:, 2]
        # mug center z
        t_em_t = T_em_t[:3, 3]
        mug_z = torch.einsum("nij,j->ni", R, t_em_t)[:, 2] + ee_z
        ee_pen = torch.clamp(z_thr - ee_z, min=0.0)
        mug_pen = torch.clamp(z_thr - mug_z, min=0.0)
        return (ee_pen ** 2).sum() + (mug_pen ** 2).sum()
    return cost


def combined_obs_cost(*cost_fns):
    """Sum multiple obs_cost_fn callables."""
    def cost(q_traj):
        total = None
        for f in cost_fns:
            c = f(q_traj)
            total = c if total is None else total + c
        return total
    return cost
