import numpy as np
import json
from functools import lru_cache
import cv2
import pickle
from tqdm import tqdm
from typing import List, Dict

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler


from kp_preprocess import get_2d3d_perspective_transform, get_points_bbox, get_points_center_scale


class HandMeshEvalDatasetHL(Dataset):
    def __init__(self, json_path, img_size=(224, 224), scale_enlarge=1.2, rot_angle=0):
        super().__init__()

        with open(json_path) as f:
            self.all_image_info = json.load(f)
        self.all_info = [{"image_path": image_path} for image_path in self.all_image_info]
        self.img_size = img_size
        self.scale_enlarge = scale_enlarge
        self.rot_angle = rot_angle

    def __len__(self):
        return len(self.all_image_info)
    
    def read_image(self, img_path):
        img = cv2.imread(img_path)
        return img
    
    def __getitem__(self, index):
        image_path = self.all_image_info[index]
        img = self.read_image(image_path)
        
        # [TODO] 390.0, 50.0, 500.0, 620.0
        # K = np.array([[767.8338012695312, 0.0, 629.979248046875], [0.0, 767.8338012695312, 348.5633239746094], [0.0, 0.0, 1.0]])
        K = np.array([[731.158, 0.0, 634.312], [0.0, 731.158, 351.922], [0.0, 0.0, 1.0]])
        # center = np.array([640.0, 360.0])
        center = np.array([630.0, 400.0])
        scale = 660.0
        
        # perspective trans
        new_K, trans_matrix_2d, trans_matrix_3d = get_2d3d_perspective_transform(K, center, scale, self.rot_angle, self.img_size[0])
        img_processed = cv2.warpPerspective(img, trans_matrix_2d, self.img_size)

        if img_processed.ndim == 2:
            img_processed = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2BGR)
        img_processed = np.transpose(img_processed, (2, 0, 1))

        return {
            "img": np.ascontiguousarray(img_processed),
            "trans_matrix_2d": trans_matrix_2d,
            "trans_matrix_3d": trans_matrix_3d,
            "center": center,
            "scale": scale,
        }
        
    def __str__(self):
        return json.dumps(len(self.all_image_info))
