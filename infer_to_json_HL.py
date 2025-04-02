
import json
import numpy as np
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

from cfg import _CONFIG
from hand_net import HandNet
from eval_dataset_HL import HandMeshEvalDatasetHL
from utils import get_log_model_dir

from scipy.linalg import orthogonal_procrustes
import open3d as o3d
import numpy as np
import cv2
import time
from typing import List
import pyrender
import trimesh
import pickle
from copy import deepcopy
from scipy.spatial.transform import Rotation as sr


def create_raymond_lights() -> List[pyrender.Node]:
    """
    Return raymond light nodes for the scene.
    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes

def vertices_to_trimesh(faces, vertices, camera_translation, mesh_base_color=(1.0, 1.0, 0.9), 
                            rot_axis=[1,0,0], rot_angle=0, is_right=1):
        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        if is_right:
            mesh = trimesh.Trimesh(deepcopy(vertices) + camera_translation, deepcopy(faces), vertex_colors=vertex_colors)
        else:
            mesh = None
            print("[ERROR] left hand not supported yet")
        
        rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), rot_axis)
        mesh.apply_transform(rot)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        return mesh

def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)

def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)

def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))

def rotx(theta):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )

def roty(theta):
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )

def rotz(theta):
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )

def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    # get lights in a circle around origin at elevation
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses

def add_point_lighting(scene, cam_node, color=np.ones(3), intensity=1.0):
    # from phalp.visualize.py_renderer import get_light_poses
    light_poses = get_light_poses(dist=0.5)
    light_poses.append(np.eye(4))
    cam_pose = scene.get_pose(cam_node)
    for i, pose in enumerate(light_poses):
        matrix = cam_pose @ pose
        node = pyrender.Node(
            name=f"plight-{i:02d}",
            light=pyrender.PointLight(color=color, intensity=intensity),
            matrix=matrix,
        )
        if scene.has_node(node):
            continue
        scene.add_node(node)

def add_lighting(scene, cam_node, color=np.ones(3), intensity=1.0):
    # from phalp.visualize.py_renderer import get_light_poses
    light_poses = get_light_poses()
    light_poses.append(np.eye(4))
    cam_pose = scene.get_pose(cam_node)
    for i, pose in enumerate(light_poses):
        matrix = cam_pose @ pose
        node = pyrender.Node(
            name=f"light-{i:02d}",
            light=pyrender.DirectionalLight(color=color, intensity=intensity),
            matrix=matrix,
        )
        if scene.has_node(node):
            continue
        scene.add_node(node)

def render_rgba(faces,
        vertices: np.array,
        rot_axis=[0,1,0],
        rot_angle=0,
        mesh_base_color=(1.0, 1.0, 0.9),
        scene_bg_color=(0,0,0),
        render_res=[256, 256],
        focal_length=731.158,
        is_right=None,
        side_view=False
    ):

    renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                            viewport_height=render_res[1],
                                            point_size=1.0)

    if side_view:
        rot_view = sr.from_euler('xyz', [0, -90, 0], degrees=True).as_matrix()
    else:
        rot_view = np.eye(3)

    # [AD-HOC] depth increased
    mesh = vertices_to_trimesh(faces, vertices @ rot_view.T + np.array([0,0,0.7]), \
                               np.array([0, 0, 0]), mesh_base_color, rot_axis, rot_angle, is_right=is_right)
    mesh = pyrender.Mesh.from_trimesh(mesh)

    scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                            ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    camera_pose = np.eye(4)
    camera_center = [render_res[0] / 2., render_res[1] / 2.]
    camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                        cx=camera_center[0], cy=camera_center[1], zfar=1e12)

    # Create camera node and add it to pyRender scene
    camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
    scene.add_node(camera_node)
    add_point_lighting(scene, camera_node)
    add_lighting(scene, camera_node)

    light_nodes = create_raymond_lights()
    for node in light_nodes:
        scene.add_node(node)

    color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    renderer.delete()

    return color


def infer_single_json(output_dir, faces, val_cfg, bmk, model, rot_angle=0):
    dataset = HandMeshEvalDatasetHL(bmk["json_dir"], val_cfg["IMAGE_SHAPE"], bmk["scale_enlarge"], rot_angle=rot_angle)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=val_cfg["BATCH_SIZE"], num_workers=4, timeout=60)
        
    HAND_WORLD_LEN = 0.2
    ROOT_INDEX = _CONFIG['DATA'].get('ROOT_INDEX', 9)
        
    pred_uv_list = []
    pred_joints_list = []
    pred_vertices_list = []

    fps = 0.0
    for cur_iter, batch_data in enumerate(tqdm(dataloader)):
        for k in batch_data:
            batch_data[k] = batch_data[k].cuda().float()
        image = batch_data['img']
        scale = batch_data['scale']
                
        trans_matrix_2d = batch_data['trans_matrix_2d']
        trans_matrix_3d = batch_data['trans_matrix_3d']
        
        trans_matrix_2d_inv = torch.linalg.inv(trans_matrix_2d)        
        trans_matrix_3d_inv = torch.linalg.inv(trans_matrix_3d)
        
        t0 = time.time()
        if cur_iter == 0:   # warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = model(image)
        with torch.no_grad():
            res = model(image)
            joints = res["joints"]
            uv = res["uv"]
            vertices = res['vertices']
        t1 = time.time()
        # print("==> inference time: ", t1-t0, ", fps: " , 1/(t1-t0))
        fps += 1.0 / (t1 - t0)
            
        vertices = vertices.reshape(-1, 778, 3)
        joints = joints.reshape(-1, 21, 3)
        uv = uv.reshape(-1, 21, 2) * val_cfg['IMAGE_SHAPE'][0]
        
        joints_root = joints[:, ROOT_INDEX][:, None, :]
        joints = joints - joints_root
        vertices = vertices - joints_root
        
        joints = (trans_matrix_3d_inv @ torch.transpose(joints, 1, 2)).transpose(1, 2) 
        vertices = (trans_matrix_3d_inv @ torch.transpose(vertices, 1, 2)).transpose(1, 2) 
        
        b, j = uv.shape[:2]
        pad = torch.ones((b, j, 1)).to(uv.device)
        uv = torch.concat([uv, pad], dim=2)        
        uv = (trans_matrix_2d_inv @ torch.transpose(uv, 1, 2)).transpose(1, 2)
        uv = uv[:, :, :2] / (uv[:, :, 2:] + 1e-7)

        pred_uv_list += uv.cpu().numpy().tolist()
        pred_joints_list += joints.cpu().numpy().tolist()
        pred_vertices_list += vertices.cpu().numpy().tolist()

        # Visualize mesh
        misc_args = dict(
                mesh_base_color=(0.1, 0.3, 1.0),
                scene_bg_color=(1, 1, 1),
                focal_length= 731.158 / val_cfg['IMAGE_SHAPE'][0] * scale,
            )
        cam_view = render_rgba(faces, \
                    vertices.cpu().numpy()[0], \
                    render_res=[scale*2, scale*2], is_right=True, **misc_args)
        # input_img = image.cpu().numpy().astype(np.float32)[:,:,::-1]/255.0
        # input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
        # input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
        no_overlay = cam_view[:,:,:3]# * cam_view[:,:,3:]
        cv2.imwrite(output_dir + '/' + 'front_view_' + dataset.all_info[cur_iter]['image_path'].split("/")[-1], 255*no_overlay[:, :, ::-1])

        # SIDE VIEW
        cam_view = render_rgba(faces, \
                    vertices.cpu().numpy()[0], \
                    render_res=[scale*2, scale*2], is_right=True, \
                    side_view=True, **misc_args)
        no_overlay = cam_view[:,:,:3]# * cam_view[:,:,3:]
        cv2.imwrite(output_dir + '/' + 'side_view_' + dataset.all_info[cur_iter]['image_path'].split("/")[-1], 255*no_overlay[:, :, ::-1])
        
    print("** Data size: ", len(pred_uv_list), ", Mean FPS: ", fps / float(len(pred_uv_list)))
    
    return pred_uv_list, pred_joints_list, pred_vertices_list


def main(epoch, tta=False, postfix=""):
    
    # MANO, for visualization
    MANO_PARAMS_PATH = "models/"
    mano_path = os.path.join(MANO_PARAMS_PATH, "MANO_RIGHT_C.pkl")
    with open(mano_path, 'rb') as mano_file:
        MANO_DATA_RIGHT = pickle.load(mano_file)
    faces_ = np.array(MANO_DATA_RIGHT["f"]).astype("int32")    
    faces = torch.from_numpy(faces_)

    val_cfg = _CONFIG['VAL']
    
    assert epoch.startswith('epoch'), "type epoch_15 for the 15th epoch"
    log_model_dir = get_log_model_dir(_CONFIG['NAME'])
    print("?: ", log_model_dir)
    model_path = os.path.join(log_model_dir, epoch)
    # from IPython import embed 
    # embed()
    # exit()
    print(model_path)
    model = HandNet(_CONFIG, pretrained=False)

    checkpoint = torch.load(open(model_path, "rb"), map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    model.cuda()
    
    bmk = val_cfg['BMK']

    # output_dir
    output_dir = os.path.join(log_model_dir, "evals", bmk['name'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dataset = HandMeshEvalDatasetHL(bmk["json_dir"], val_cfg["IMAGE_SHAPE"], bmk["scale_enlarge"])

    pred_uv_list, xyz_pred_list, verts_pred_list = infer_single_json(output_dir, faces, val_cfg, bmk, model, rot_angle=0)
    
    joints3d_stack = []
    joints2d_stack = []
    for pred_uv, pred_xyz, pred_vertices, ori_info in zip(pred_uv_list, xyz_pred_list, verts_pred_list, dataset.all_info):
        ori_info['pred_uv'] = pred_uv
        ori_info['pred_xyz'] = pred_xyz
        ori_info['pred_vertices'] = pred_vertices
        
        joints3d_now = []
        for joint3d in pred_xyz:
            joints3d_now += joint3d
        joints3d_stack += [joints3d_now]
        joints2d_now = []
        for joint2d in pred_uv:
            joints2d_now += joint2d
        joints2d_stack += [joints2d_now]

        img_ = cv2.imread(ori_info['image_path'])
        for kp in pred_uv:
            img_ = cv2.circle(img_, (int(kp[0]), int(kp[1])), 3, (0,0,255), 3)
        cv2.imwrite(os.path.join(output_dir, ori_info['image_path'].split("/")[-1]), img_)

    result_json_path = os.path.join(output_dir, f"{epoch}{postfix}.json")
    with open(result_json_path, 'w') as f:
        json.dump(dataset.all_info, f)
        print(f"Result save to {result_json_path}")
        
    with open(output_dir + "/pred_joints3d.json", 'w') as f:
        json.dump(joints3d_stack, f)
        print(f"XYZ save to {output_dir}")
    with open(output_dir + "/pred_joints2d_img.json", 'w') as f:
        json.dump(joints2d_stack, f)
        print(f"UV save to {output_dir}")
        
if __name__ == "__main__":
    from fire import Fire
    Fire(main)

