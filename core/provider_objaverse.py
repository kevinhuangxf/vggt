import os
import cv2
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import PIL
import trimesh
from PIL import Image

import kiui
from kiui.cam import orbit_camera, OrbitCamera

from core.options import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# import objaverse
# import pandas as pd
# import open3d as o3d
# import trimesh
import json

# objaverse.BASE_PATH = '/workspace/datasets/objaverse'
# objaverse._VERSIONED_PATH = '/workspace/datasets/objaverse/hf-objaverse-v1'

# kiui_uids = pd.read_csv("/workspace/code/objaverse_filter/kiuisobj_v1_merged_80K.csv", header=None)
# uids = kiui_uids[1].values.tolist()

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

# def normalize_point_cloud(point_cloud: torch.Tensor, box_scale: float):
#     # Assuming point_cloud is a tensor of shape (N, 3) where N is the number of points
#     bbox_min, bbox_max = torch.min(point_cloud, dim=0)[0], torch.max(point_cloud, dim=0)[0]
#     scale = box_scale / torch.max(bbox_max - bbox_min)
#     # Scale the point cloud
#     point_cloud *= scale
#     # Recompute the bounding box
#     bbox_min, bbox_max = torch.min(point_cloud, dim=0)[0], torch.max(point_cloud, dim=0)[0]
#     offset = -(bbox_min + bbox_max) / 2
#     # Translate the point cloud
#     point_cloud += offset
#     return point_cloud


def normalize_point_cloud(point_cloud, box_scale):
    # Assuming point_cloud is a tensor of shape (N, 3) where N is the number of points
    bbox_min, bbox_max = np.min(point_cloud, axis=0), np.max(point_cloud, axis=0)
    scale = box_scale / np.max(bbox_max - bbox_min)
    # Scale the point cloud
    point_cloud *= scale
    # Recompute the bounding box
    bbox_min, bbox_max = np.min(point_cloud, axis=0), np.max(point_cloud, axis=0)
    offset = -(bbox_min + bbox_max) / 2
    # Translate the point cloud
    point_cloud += offset

    return point_cloud


def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    # Mask for valid coordinates
    valid_mask = (depthmap > 0.0)
    # valid_mask = (depthmap >= 0.0)
    return X_cam, valid_mask


def depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, **kw):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels."""
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)

    X_world = X_cam # default
    if camera_pose is not None:
        # R_cam2world = np.float32(camera_params["R_cam2world"])
        # t_cam2world = np.float32(camera_params["t_cam2world"]).squeeze()
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        # Express in absolute coordinates (invalid depth values)
        X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]

    return X_world, valid_mask


class ObjaverseDataset(Dataset):

    def _warn(self):
        raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)')

    def __init__(self, opt: Options, training=True):
        
        self.opt = opt
        self.training = training

        # TODO: remove this barrier
        # self._warn()

        # TODO: load the list of objects for training
        # self.items = []
        # with open('TODO: file containing the list', 'r') as f:
        #     for line in f.readlines():
        #         self.items.append(line.strip())
        # self.items = uids

        # if self.training:
        #     local_views_path_json = self.opt.local_views_path_json
        # else:
        #     local_views_path_json = self.opt.eval_views_path_json
        
        local_views_path_json = self.opt.local_views_path_json

        with open(local_views_path_json, 'r') as f:
            local_views = json.load(f)
        self.items = local_views
        if len(self.items) < 2000 and self.training:
            self.items = self.items * (1000 // len(self.items))

        # local_models_path_json = self.opt.local_models_path_json
        # if local_models_path_json is not None:
        #     with open(local_models_path_json, 'r') as f:
        #         local_models_path = json.load(f)
        #     self.glb_list = local_models_path
        #     if len(self.glb_list) < 1000:
        #         self.glb_list = self.glb_list * (1000 // len(self.glb_list))

        # local_views_path_json= "/workspace/code/objaverse-rendering/valid_views.json"
        # # local_views_path_json= "/workspace/code/objaverse-rendering/valid_views_miku.json"
        # with open(local_views_path_json, 'r') as f:
        #     local_views = json.load(f)
        # self.items = local_views#[:2000]*4
        # # self.items = ["/workspace/code/objaverse-rendering/results_random/hatsune_miku/"]*1000

        # local_models_path_json= "/workspace/code/objaverse-rendering/valid_views_glb.json"
        # # local_models_path_json= "/workspace/code/objaverse-rendering/local_models_path_miku.json"
        # with open(local_models_path_json, 'r') as f:
        #     local_models_path = json.load(f)
        # self.glb_list = local_models_path#[:2000]*4
        # # self.glb_list = ["/workspace/code/objaverse-rendering/results/hatsune_miku/hatsune_miku.glb"]*1000

        # naive split
        if self.training:
            self.items = self.items[:-self.opt.batch_size]
        else:
            self.items = self.items[-self.opt.batch_size:]
        
        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1        

        # camera parameters
        self.camera_params = OrbitCamera(self.opt.input_size, self.opt.input_size, r=1.5, fovy=67.38)
        self.intrinsics = torch.from_numpy(self.camera_params.intrinsics) # np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    def __len__(self):
        return len(self.items)

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        pil_img = Image.open(path)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + color * (1 - alpha)

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha

    def load_glb_o3d(self, glb_path):
        # Load the GLB file using open3d
        mesh = o3d.io.read_triangle_mesh(glb_path)

        # Get the point positions
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=10000)

        # Get the point positions from the point cloud
        point_positions_tensor =  torch.tensor(pcd.points).float()
        point_positions_tensor[:, [1, 2]] = point_positions_tensor[:, [2, 1]]
        # point_positions_tensor = point_positions_tensor / (torch.abs(point_positions_tensor).max())
        point_positions_tensor = normalize_point_cloud(point_positions_tensor, 2.0)
        # self.glb_list.append(point_positions_tensor)
        return point_positions_tensor

    def load_glb(self, glb_path, num_sample_points=65536):
        # Load the .glb file
        # scene_or_mesh = trimesh.load_mesh(os.path.join(glb_path, os.path.basename(os.path.dirname(glb_path)) + '.glb'))
        scene_or_mesh = trimesh.load_mesh(glb_path)

        # Check if the loaded object is a Scene
        if isinstance(scene_or_mesh, trimesh.Scene):
            # Combine all the meshes in the scene into a single mesh
            mesh = scene_or_mesh.dump(concatenate=True)
        else:
            # The loaded object is already a Mesh
            mesh = scene_or_mesh

        # Sample points from the mesh
        # points = mesh.sample(10000)
        points = trimesh.sample.sample_surface(mesh, num_sample_points)

        point_positions_tensor =  torch.tensor(points[0]).float()
        point_positions_tensor = normalize_point_cloud(point_positions_tensor, 2.0)
        return point_positions_tensor

    def load_depth(self, depth_path):
        dep_img = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if dep_img is None:
            print(f'[WARN] failed to load depth image {depth_path}')
        dep_img[dep_img > 100] = 0
        if dep_img.max() - dep_img.min() == 0:
            dep_img = np.zeros_like(dep_img)
        else:
            dep_img = (dep_img - dep_img.min())/(dep_img.max() - dep_img.min())
        dep_img = dep_img[:, :, 0:1]

        dep_img = torch.from_numpy(dep_img).permute(2, 0, 1).contiguous().float()
        return dep_img

    def __getitem__(self, idx):

        uid = self.items[idx]
        results = {}

        # load num_views images
        images = []
        masks = []
        cam_poses = []
        depths = []
        
        vid_cnt = 0

        # # load glb
        # if hasattr(self.opt, "num_sample_points"):
        #     num_sample_points = self.opt.num_sample_points
        # else:
        #     num_sample_points = 32768
        # point_cloud = self.load_glb(self.glb_list[idx], num_sample_points)

        # TODO: choose views, based on your rendering settings
        # if self.training:
        #     # input views are in (36, 72), other views are randomly selected
        #     vids = np.random.permutation(np.arange(36, 73))[:self.opt.num_input_views].tolist() + np.random.permutation(100).tolist()
        # else:
        #     # fixed views
        #     vids = np.arange(36, 73, 4).tolist() + np.arange(100).tolist()
        # vids = np.arange(0, 8, 1).tolist()
        vids = np.random.permutation(np.arange(32))[:self.opt.num_views].tolist() # self.opt.num_input_views

        bkg_color = [1.0, 1.0, 1.0]

        # load intrinsics
        intrinsics_path = os.path.join(uid, f'intrinsics.npy')
        intrinsics_np = np.load(intrinsics_path)

        intrinsics = torch.zeros(3, 3)
        intrinsics[0, 0] = intrinsics_np[0][0] # self.intrinsics[0]
        intrinsics[1, 1] = intrinsics_np[0][1] # self.intrinsics[1]
        intrinsics[0, 2] = intrinsics_np[1][0] # self.intrinsics[2]
        intrinsics[1, 2] = intrinsics_np[1][1] # self.intrinsics[3]
        intrinsics[2, 2] = 1

        depth_path_list = []

        for vid in vids:

            image_path = os.path.join(uid, 'rgba', f'{vid:03d}.png')
            camera_path = os.path.join(uid, 'pose', f'{vid:03d}.npy')
            if self.opt.use_depth:
                depth_path = os.path.join(uid, 'depth', f'{vid:03d}_depth0001.exr')
                depth_path_list.append(depth_path)
                depth_img = self.load_depth(depth_path)
            try:
                # TODO: load data (modify self.client here)
                # image = np.frombuffer(self.client.get(image_path), np.uint8)
                # image = torch.from_numpy(cv2.imdecode(image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]
                image, alpha = self.load_im(image_path, bkg_color)
                # c2w = [float(t) for t in self.client.get(camera_path).decode().strip().split(' ')]
                # c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)
                
                c2w = np.load(camera_path)
                c2w = np.concatenate([c2w, np.array([[0, 0, 0, 1]])], axis=0)

                # blender world + opencv cam --> opengl world & cam
                c2w[1] *= -1
                c2w[[1, 2]] = c2w[[2, 1]]

                c2w = torch.from_numpy(c2w).float()
            except Exception as e:
                print(f'[WARN] dataset {uid} {vid}: {e}')
                continue
            
            # c2w = torch.from_numpy(orbit_camera(0, 360 - vid * 90, radius=2.0, opengl=True))

            # TODO: you may have a different camera system
            # c2w = torch.linalg.inv(w2c)
            # blender world + opencv cam --> opengl world & cam
            # c2w[2] *= -1
            # c2w[[1, 2]] = c2w[[2, 1]]
            # c2w[:3, 1:3] *= -1 # invert up and forward direction

            # c2w_new = c2w.clone()
            # c2w_new[1] = c2w[2]
            # c2w_new[2] = c2w[1]
            # c2w = c2w_new

            # scale up radius to fully use the [-1, 1]^3 space!
            # c2w[:3, 3] *= self.opt.cam_radius / 1.5 # 1.5 is the default scale

            # # depth to point cloud
            # c2w[:3, 1:3] *= -1 # invert up & forward direction
            
            # img = Image.open(image_path).convert('RGB')
            # h, w = (img.size[0], img.size[1])
            # img = np.array(img)
            # rgb_image = img

            # resolution = [h, w] # [320, 320]
            # depthmap_origin = depth_img.numpy()[0]
            # # rgb_image, depthmap, intrinsics = crop_resize_if_necessary(
            # #     rgb_image, depthmap_origin, intrinsics_origin, resolution
            # # )

            # pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(depthmap_origin, intrinsics_origin, c2w.numpy())
            # # pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(depthmap, intrinsics, c2w.numpy())
            # pts = np.concatenate([p[m] for p, m in zip(pts3d, valid_mask)])
            # col = np.concatenate([p[m] for p, m in zip(img, valid_mask)])

            # pts = normalize_point_cloud(pts.reshape(-1, 3), 2.0)
            # col = col.reshape(-1, 3)
            # pts_list.append(pts)
            # col_list.append(col)

            images.append(image)
            masks.append(alpha.squeeze(0))
            cam_poses.append(c2w)
            if self.opt.use_depth:
                depths.append(depth_img)

            vid_cnt += 1
            if vid_cnt == self.opt.num_views:
                break

        if vid_cnt < self.opt.num_views:
            print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
            n = self.opt.num_views - vid_cnt
            images = images + [images[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n
          
        images = torch.stack(images, dim=0) # [V, C, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]
        if self.opt.use_depth:
            depths = torch.stack(depths, dim=0) # [V, 4, 4]
        # # export point cloud
        # pts = np.concatenate(pts_list)
        # col = np.concatenate(col_list)

        # pct = trimesh.PointCloud(pts, colors=col)
        # pct = trimesh.PointCloud(pts, colors=col.reshape(-1, 3))
        # pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        # pct = trimesh.PointCloud(pts.reshape(-1, 3))

        # # Save to a PLY file
        # print('saving to local file.')
        # pct.export('point_cloud.ply')
        # print('saving to local file done.')
        # normalized camera feats as in paper (transform the first pose to a fixed position)
        # transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[1])
        # cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

        # cam_poses[0, 1, 1] = cam_poses[0, 1, 1] * -1

        # elevation = 0
        # cam_poses = []
        # for azi in [0, 90, 180, 270, 0, 90, 180, 270]:
        #     cam_pose = torch.from_numpy(orbit_camera(elevation, azi, radius=1.5, opengl=True)).unsqueeze(0)
        #     cam_pose[:, :3, 1:3] *= -1 # invert up & forward direction
        #     cam_poses.append(cam_pose[0])
        # cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        # data augmentation
        if self.training:
            # apply random grid distortion to simulate 3D inconsistency
            if random.random() < self.opt.prob_grid_distortion:
                images_input[1:] = grid_distortion(images_input[1:])
            # apply camera jittering (only to input!)
            if random.random() < self.opt.prob_cam_jitter:
                cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # resize render ground-truth images, range still in [0, 1]
        if self.opt.use_depth:
            results['depths_output'] = F.interpolate(depths, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

        # build rays for input views
        rays_embeddings = []
        for i in range(self.opt.num_input_views):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)
     
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
        results['input'] = final_input
        # results['input'] = images_input

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        intrinsics = torch.zeros(3, 3)
        intrinsics[0, 0] = self.intrinsics[0]
        intrinsics[1, 1] = self.intrinsics[1]
        intrinsics[0, 2] = self.intrinsics[2]
        intrinsics[2, 2] = self.intrinsics[3]
        results['intrinsics'] = intrinsics.unsqueeze(0).repeat(self.opt.num_views, 1, 1)
        results['extrinsics'] = cam_poses

        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        # results['point_cloud'] = point_cloud

        return results
