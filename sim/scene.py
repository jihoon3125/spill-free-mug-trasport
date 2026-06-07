"""Sapien scene: table + UR5e + xhand + mug rigidly held at grasp config.

Loads:
  data/grasp_info.txt — relative pose + qpos (18 = 6 UR5e + 12 xhand)
  data/mug_mesh.obj   — mug geometry (scaled by 0.12)

The scene places UR5e+xhand with the recorded qpos. The mug is positioned at
'Relative Object Pose (Base-to-Object)' from grasp_info.txt — that is, where
the mug sits in the robot base frame when the hand is at qpos.

For motion planning, EE pose is derived from FK(qpos[:6]) and the mug pose
is rigidly attached to EE via T_ee_to_mug (precomputed once from grasp_info).

Run:
    python -m sim.scene             # headless, saves figs/scene_init.png
    python -m sim.scene --gui       # opens viewer
"""
from __future__ import annotations
import re, argparse
from pathlib import Path
import numpy as np
import sapien

ROOT = Path(__file__).resolve().parents[1]
THESIS = Path("/home/dongjae/UltraDexGrasp/UltraDexGrasp_jihoon")
URDF = THESIS / "asset/ur5e_with_xhand_urdf_offset_sim2real/ur5e_with_xhand_left_limited_joint_sapien.urdf"

# Joint order in the URDF (verified by xml parsing)
JOINT_ORDER = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    "left_hand_thumb_bend_joint", "left_hand_thumb_rota_joint1", "left_hand_thumb_rota_joint2",
    "left_hand_index_bend_joint", "left_hand_index_joint1", "left_hand_index_joint2",
    "left_hand_mid_joint1", "left_hand_mid_joint2",
    "left_hand_ring_joint1", "left_hand_ring_joint2",
    "left_hand_pinky_joint1", "left_hand_pinky_joint2",
]


def parse_grasp_info(path: Path) -> dict:
    txt = path.read_text()
    scale = float(re.search(r'Object Scale:\s*([\d.]+)', txt).group(1))
    m_obj_t = re.search(r'Relative Object Pose.*?Translation.*?\[(.*?)\]', txt, re.S)
    m_obj_q = re.search(r'Relative Object Pose.*?Rotation.*?\[(.*?)\]', txt, re.S)
    m_qpos = re.search(r'Joint Angles.*?\[(.*?)\]', txt, re.S)
    obj_t = np.array(list(map(float, m_obj_t.group(1).split(','))))
    obj_q = np.array(list(map(float, m_obj_q.group(1).split(','))))   # wxyz
    qpos = np.array(list(map(float, m_qpos.group(1).split(','))))
    return dict(scale=scale, obj_pos=obj_t, obj_quat_wxyz=obj_q, qpos=qpos)


def build_scene(gui: bool = False):
    g = parse_grasp_info(ROOT / "data/grasp_info.txt")
    print(f"[load] mug scale={g['scale']}  qpos.shape={g['qpos'].shape}")

    scene = sapien.Scene()
    scene.set_timestep(1.0 / 240.0)
    scene.add_ground(0.0)
    scene.set_ambient_light([0.4, 0.4, 0.4])
    scene.add_directional_light([0, 0.5, -1], [0.9, 0.9, 0.9])

    # Robot (UR5e + xhand) — fixed base at origin
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(str(URDF))
    robot.set_root_pose(sapien.Pose([0, 0, 0]))

    # Set qpos in URDF joint order
    active_joints = {j.name: j for j in robot.get_active_joints()}
    print(f"[robot] active joints in built articulation: {len(active_joints)}")
    qpos_ordered = np.zeros(len(active_joints))
    # Map: sapien returns joints in its own order; align via JOINT_ORDER
    sapien_order = [j.name for j in robot.get_active_joints()]
    name_to_grasp_idx = {n: i for i, n in enumerate(JOINT_ORDER)}
    for s_idx, name in enumerate(sapien_order):
        if name in name_to_grasp_idx:
            qpos_ordered[s_idx] = g['qpos'][name_to_grasp_idx[name]]
        else:
            print(f"[warn] sapien joint not in expected order: {name}")
    robot.set_qpos(qpos_ordered)

    # Mug — rigid body with mesh collision + visual, kinematic for now
    builder = scene.create_actor_builder()
    builder.add_convex_collision_from_file(
        str(ROOT / "data/mug_mesh.obj"), scale=[g['scale']] * 3)
    builder.add_visual_from_file(
        str(ROOT / "data/mug_mesh.obj"), scale=[g['scale']] * 3)
    mug = builder.build_kinematic(name="mug")
    mug.set_pose(sapien.Pose(g['obj_pos'].tolist(), g['obj_quat_wxyz'].tolist()))

    # Simple table (visual only)
    table_b = scene.create_actor_builder()
    table_b.add_box_visual(half_size=[0.5, 0.5, 0.01], material=sapien.render.RenderMaterial(base_color=[0.7, 0.6, 0.5, 1]))
    table = table_b.build_kinematic(name="table")
    table.set_pose(sapien.Pose([0.6, 0, 0.0]))

    return scene, robot, mug, g


def render_headless(scene, save_path: Path):
    cam = scene.add_camera("cam", 1280, 720, fovy=1.0, near=0.05, far=10.0)
    # Look at mug location from a 45° angle
    cam.set_pose(sapien.Pose(p=[1.2, -0.8, 0.9], q=look_at_quat([1.2, -0.8, 0.9], [0.6, -0.2, 0.4])))
    scene.update_render()
    cam.take_picture()
    rgb = cam.get_picture("Color")[..., :3]    # uint8 HxWx3
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    import PIL.Image
    PIL.Image.fromarray(rgb).save(save_path)
    print(f"[render] saved {save_path}")


def look_at_quat(eye, target, up=(0, 0, 1)):
    """Sapien camera convention: -Z forward, +Y up.
       Returns quaternion wxyz."""
    eye = np.array(eye, dtype=float)
    target = np.array(target, dtype=float)
    up = np.array(up, dtype=float)
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    # Sapien camera looks along +X (different convention from OpenGL).
    # Build rotation: col0=f, col1=-r, col2=u  (camera frame: X-forward, Y-left, Z-up)
    R = np.column_stack([f, -r, u])
    from scipy.spatial.transform import Rotation as Rot
    q_xyzw = Rot.from_matrix(R).as_quat()
    return [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "figs/scene_init.png"))
    a = ap.parse_args()

    scene, robot, mug, g = build_scene(gui=a.gui)
    ee_pose = None
    for link in robot.get_links():
        if link.name == "base":
            ee_pose = link.pose
            break
    print(f"[ee] base link pose: {ee_pose}")
    print(f"[mug] pose: {mug.pose}")

    if a.gui:
        viewer = scene.create_viewer()
        viewer.set_camera_xyz(1.2, -0.8, 0.9)
        viewer.set_camera_rpy(0, -0.3, 2.0)
        while not viewer.closed:
            scene.step()
            scene.update_render()
            viewer.render()
    else:
        render_headless(scene, Path(a.out))


if __name__ == "__main__":
    main()
