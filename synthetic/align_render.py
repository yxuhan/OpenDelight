import os
import random
import cv2
from tqdm import tqdm
import trimesh
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms

from mesh_renderer import MeshRenderer
from align_utils import align_face
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_root', type=str, default="workspace/ava256_raw/fold0/20210810--1306--FXN596_029693")
parser.add_argument('--scan_root', type=str, default="data/ava256_4k/20210810--1306--FXN596_029693")
opt, _ = parser.parse_known_args()


LDM_INDEX = np.array([
    # contour 17
    2053, 1593, 1727, 729, 940, 839, 1421, 824, 1492, 3068, 2926, 3085, 3186, 2970, 3967, 3822, 4277, 
    # eyebrow 10
    460, 835, 534, 2170, 2146, 4365, 4384, 2768, 2754, 2697, 
    # nose 9
    1553, 4537, 4798, 4782, 97, 2217, 1673, 4424, 2347, 
    # left eye 6
    1573, 1288, 1201, 1141, 1068, 1469, 
    # right eye 6
    4163, 3376, 3541, 3802, 3306, 3322, 
    # mouth outer 12
    5182, 5229, 2222, 5399, 4429, 5758, 5743, 5887, 5964, 5481, 5448, 5365, 
    # mouth inner 8
    5179, 5250, 5325, 5778, 5732, 5943, 5480, 5426, 
])


def draw_all_lmk(img, lmks):
    img_copy = img.copy()
    for lmk in lmks:
        x = int(lmk[0] + 0.5)
        y = int(lmk[1] + 0.5)
        cv2.circle(img_copy, (x, y), 1, (0, 255, 0), -1)
    return img_copy


def torch_to_cv2(img):
    img = img.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255).astype(np.uint8)
    return img


def align_face_mx(img_cv2, output_size, ldm):
    img_float = img_cv2.astype('float32')
    aligned_face, H = align_face(
		img_float, 
		output_size=output_size, 
		lm=ldm,
        border_scale=0.85,
	)
    aligned_face = aligned_face.astype('uint8')
    return aligned_face


class GBufferRenderer:
    def __init__(self):
        self.device = "cuda"
        self.img_size = 1024
        self.mesh_renderer = MeshRenderer(self.device)
    
    def load_geometry(self, mesh_uv_path, landmark_path):
        device = self.device
        mesh = trimesh.load_mesh(mesh_uv_path, process=False)
        uv = torch.from_numpy(mesh.visual.uv).to(device).float()  # [v,2]
        
        vertices = torch.from_numpy(mesh.vertices).to(device).float()  # [v,3]
        rot_mat = torch.from_numpy(
            Rotation.from_euler("x", 180, degrees=True).as_matrix()
        ).to(device).float()
        vertices = (rot_mat @ vertices[..., None])[..., 0]

        # ldm = torch.load(landmark_path, map_location=self.device)
        # self.ldm = (rot_mat @ ldm[..., None])[..., 0]

        normal = torch.from_numpy(mesh.vertex_normals).to(device).float()  # [v,3]
        normal = F.normalize(normal, dim=-1)
        faces = torch.from_numpy(mesh.faces).to(device)  # [f,3]
        self.mesh = mesh
        self.uv = uv[None, ...]  # [1,v,2]
        self.uv = 2 * self.uv - 1
        self.uv[..., 1] *= -1
        self.vertices = vertices[None, ...]  # [1,v,3]
        self.faces = faces[None, ...]  # [1,v,3]

        self.ldm = vertices[LDM_INDEX]  # [68,3]
    
    def load_img(self, img_path):
        return transforms.ToTensor()(
            Image.open(img_path)
        ).to(self.device)[None, :3, ...]  # [1,3,h,w]
    
    def load_camera_config(self, cam_info):
        # cam_info_path = os.path.join(render_root, "info.npy")
        # cam_info = np.load(cam_info_path, allow_pickle=True).item()
        cam_int_list = []
        cam_ext_list = []
        for k in sorted(cam_info.keys()):
            cur_fov = cam_info[k]["cam"]["fov"] / np.pi * 180
            cur_cam_trans = cam_info[k]["cam"]["trans"]
            cur_cam_rot = cam_info[k]["cam"]["rot"]
            cur_cam_int = self.fov_to_cam_int(cur_fov)

            cur_cam_ext = np.eye(4)
            cur_cam_ext[:3, :3] = cur_cam_rot
            cur_cam_ext[:3, 3] = cur_cam_trans
            cur_cam_ext = torch.from_numpy(cur_cam_ext).to(self.device).float()
            
            cam_int_list.append(cur_cam_int)
            cam_ext_list.append(cur_cam_ext)
        
        cam_int = torch.cat(cam_int_list, dim=0)
        cam_ext = torch.stack(cam_ext_list, dim=0)

        cam_ext = torch.inverse(cam_ext)[:, :3]
        return cam_int, cam_ext

    def fov_to_cam_int(self, fov):
        fov = fov / 180 * np.pi
        focal = 0.5 / np.tan(0.5 * fov)
        cam_int = torch.tensor([
            [focal, 0, 0.5],
            [0, focal, 0.5],
            [0, 0, 1],
        ], device=self.device).float()
        return cam_int[None, ...].to(self.device)

    def proj_3d_points(self, ldm, cam_int, cam_ext):
        # project all the vertices to screen
        rot = cam_ext[:, :3, :3]  # [b,3,3]
        trans = cam_ext[:, :3, 3:]  # [b,3,1]
        ldm = (rot @ ldm[..., None] + trans)[..., 0]
        
        ldm = (cam_int @ ldm[..., None])[..., 0]
        ldm = ldm[..., :2] / ldm[..., 2:]
        ldm = ldm * self.img_size
        return ldm

    def render_a_image(self):
        render_root = opt.data_root
        render_info = np.load(
            os.path.join(render_root, "info.npy"), allow_pickle=True
        ).item()

        mesh_path = render_info["scene"]["obj_path"]
        ref_map_root = render_info["scene"]["export_dir"]

        # old dirs
        render_img_root = os.path.join(render_root, "image")
        diff_img_root = os.path.join(render_root, "diffuse")
        render_spec_root = os.path.join(render_root, "image_wo_spec")
        normal_img_root = os.path.join(render_root, "normal")

        # new dirs
        align_img_root = os.path.join(render_root, "align_image")
        align_diff_root = os.path.join(render_root, "align_diffuse")
        mask_img_root = os.path.join(render_root, "mask")
        os.makedirs(align_img_root, exist_ok=True)
        os.makedirs(mask_img_root, exist_ok=True)
        os.makedirs(align_diff_root, exist_ok=True)

        self.load_geometry(mesh_path, None)

        attrs = torch.cat([
            self.uv, torch.ones_like(self.vertices)
        ], dim=-1)

        common_mask = self.load_img("assets/face_mask.png")
        uv_mask = common_mask * self.load_img(os.path.join(opt.scan_root, "skin_mask_vis.png"))

        mesh_dict = {
            "faces": self.faces,
            "vertice": self.vertices,  # [1,v,3]
            "attributes": attrs,  # [1,v,3]
            "size": (self.img_size, self.img_size),
        }

        cam_int, cam_ext = self.load_camera_config(render_info["view"])
        img_list = []

        for i in tqdm(range(len(cam_ext))):
            output, pix_to_face = self.mesh_renderer.render_mesh(
                mesh_dict, cam_int[i:i+1], cam_ext[i:i+1]
            )
            ldm = self.proj_3d_points(self.ldm, cam_int[i:i+1], cam_ext[i:i+1])  # [68,2]

            mask = output[:, 2:5]
            uv = output[:, :2].permute(0, 2, 3, 1)  # [1,h,w,2]
            render_mask = F.grid_sample(uv_mask, uv, align_corners=True)[0]  # [3,h,w]
            mask = render_mask * mask
            save_image(mask, os.path.join(mask_img_root, "%05d.png" % i))

            diff_img_cv2 = cv2.imread(os.path.join(diff_img_root, "%05d.png" % i))
            diff_img_align = align_face_mx(diff_img_cv2, 512, ldm.cpu().numpy())
            cv2.imwrite(os.path.join(align_diff_root, "%05d.png" % i), diff_img_align)

            mask_img_cv2 = torch_to_cv2(mask[0])
            mask_img_align = align_face_mx(mask_img_cv2, 512, ldm.cpu().numpy())
            cv2.imwrite(os.path.join(mask_img_root, "%05d.png" % i), mask_img_align)
            
            render_img_cv2 = cv2.imread(os.path.join(render_img_root, "%05d.png" % i))
            render_img_align = align_face_mx(render_img_cv2, 512, ldm.cpu().numpy())
            cv2.imwrite(os.path.join(align_img_root, "%05d.png" % i), render_img_align)


if __name__ == "__main__":
    gbuffer_renderer = GBufferRenderer()
    gbuffer_renderer.render_a_image()
