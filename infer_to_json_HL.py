
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


def infer_single_json(val_cfg, bmk, model, rot_angle=0):
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
    print("** Data size: ", len(pred_uv_list), ", Mean FPS: ", fps / float(len(pred_uv_list)))
    
    return pred_uv_list, pred_joints_list, pred_vertices_list


def main(epoch, tta=False, postfix=""):

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

    pred_uv_list, xyz_pred_list, verts_pred_list = infer_single_json(val_cfg, bmk, model, rot_angle=0)
    
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
        cv2.imwrite(os.path.join(output_dir, ori_info['image_path'][29:]), img_)

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

