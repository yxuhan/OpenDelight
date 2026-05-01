import blenderproc as bproc
import os
import bpy
import numpy as np
import cv2
import argparse
from blenderproc.python.utility.Utility import Utility
import random
from scipy.spatial.transform import Rotation


bproc.init()


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_root', type=str, default="../data/ava256_4k/20210810--1306--FXN596_029693")
parser.add_argument('--save_root', type=str, default="../workspace/debug")
parser.add_argument('--num_view', type=int, default=-1)
parser.add_argument('--fov_cfg', type=str, default="normal")
parser.add_argument('--sss', type=int, default=1)


opt, _ = parser.parse_known_args()


HDRI_ROOT = "../data/HDRI"
HIGH_FREQ_PROB = 5
MED_FREQ_PROB = 3
LOW_FREQ_PROB = 1


def batch_camera_look_at_rotation(eye, center, up):
    if not isinstance(eye, np.ndarray):
        eye = np.array(eye, dtype=np.float32)
    if not isinstance(center, np.ndarray):
        center = np.array(center, dtype=np.float32)
    if not isinstance(up, np.ndarray):
        up = np.array(up, dtype=np.float32)
    
    b = eye.shape[0]
    
    if center.ndim == 1:
        center = np.tile(center, (b, 1))
    if up.ndim == 1:
        up = np.tile(up, (b, 1))
    
    z_axis = eye - center
    z_axis = z_axis / np.linalg.norm(z_axis, axis=1, keepdims=True)
    
    x_axis = np.cross(up, z_axis, axis=1)
    x_axis = x_axis / np.linalg.norm(x_axis, axis=1, keepdims=True)
    
    y_axis = np.cross(z_axis, x_axis, axis=1)
    
    rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=1)
    
    return rotation_matrix


class CameraRig:
    def __init__(self):
        
        pitch_angle = 35
        yaw_angle = 60
        yaw_num = 12
        pitch_num = 7
        
        pitch_angle_list = np.linspace(-pitch_angle, pitch_angle, pitch_num)
        yaw_angle_list = np.linspace(-yaw_angle, yaw_angle, yaw_num)
        
        self.verts = []
        for pitch in pitch_angle_list:
            for yaw in yaw_angle_list:
                theta = np.radians(yaw)
                phi = np.radians(pitch)
                x = np.cos(phi) * np.sin(theta)
                y = np.sin(phi)
                z = -np.cos(phi) * np.cos(theta)
                cur_vert = np.array([x, y, z], dtype=np.float32)
                self.verts.append(cur_vert)
        self.verts = np.stack(self.verts, axis=0)  # [b,3]

    def get_camera_poses(self):
        cam_pose = batch_camera_look_at_rotation(
            eye=self.verts,
            center=np.zeros(3),
            up=np.array([0, -1, 0], dtype=np.float32)
        )
        
        trans = Rotation.from_euler("x", 180, degrees=True).as_matrix()
        cam_pose = np.matmul(cam_pose.transpose(0, 2, 1), trans.T)
        
        cam_pose = np.concatenate([
            cam_pose, self.verts[..., np.newaxis]
        ], axis=-1)  # [b,3,4]
        
        bottom = np.tile(np.array([0, 0, 0, 1], dtype=np.float32).reshape(1, 1, 4), 
                         (cam_pose.shape[0], 1, 1))
        cam_pose = np.concatenate([cam_pose, bottom], axis=1)  # [b,4,4]
        
        return cam_pose


class RenderConfig:
    def __init__(self):
        self._make_light_freq_prob()
        self.config_view()
        self.config_cam()
        self.config_light()
    
    def _make_light_freq_prob(self):
        total = HIGH_FREQ_PROB + MED_FREQ_PROB + LOW_FREQ_PROB
        self.high_freq_prob = HIGH_FREQ_PROB / total
        self.med_freq_prob = MED_FREQ_PROB / total
        self.low_freq_prob = LOW_FREQ_PROB / total
    
    def _sample_light_type(self):
        rand_val = random.random()
        if rand_val < self.high_freq_prob:
            return "high"
        elif rand_val < self.high_freq_prob + self.med_freq_prob:
            return "med"
        else:
            return "low"
    
    def config_view(self):
        camera_rig = CameraRig()
        # camera_rig = CameraRigH()
        # camera_rig = CameraRigV()
        cam_poses = camera_rig.get_camera_poses()
        if opt.num_view == -1:
            select_cam_poses = cam_poses
        else:
            select_cam_poses = cam_poses[
                random.choices(
                    list(range(len(cam_poses))), 
                    k=opt.num_view,
                )
            ]
        self.num_view = len(select_cam_poses)
        self.select_cam_poses = select_cam_poses

    def config_cam(self):
        cam_dict = {
            "short": {
                "fov": 61.9275,
                "dist": 0.25,
            },
            "normal": {
                "fov": 39.5978,
                "dist": 0.4,
            },
            "long": {
                "fov": 16.3885,
                "dist": 0.95,
            },
        }

        fov_type = []
        if "short" in opt.fov_cfg:
            fov_type.append("short")
        if "normal" in opt.fov_cfg:
            fov_type.append("normal")
        if "long" in opt.fov_cfg:
            fov_type.append("long")
        select_fov_type = random.choices(fov_type, k=self.num_view)
        
        self.select_cam_rot = []
        self.select_cam_trans = []
        self.select_cam_int = []
        self.select_cam_rot_mat = []
        for i in range(self.num_view):
            cur_fov = cam_dict[select_fov_type[i]]["fov"]
            cur_dist = cam_dict[select_fov_type[i]]["dist"]
            
            cur_cam_pose = self.select_cam_poses[i]
            cur_cam_trans = cur_cam_pose[:3, 3] * cur_dist
            cur_cam_rot = cur_cam_pose[:3, :3]
            self.select_cam_rot_mat.append(cur_cam_rot)

            cur_cam_euler = Rotation.from_matrix(cur_cam_rot).as_euler("xyz", degrees=False)
            cur_cam_euler[0] += np.pi

            self.select_cam_trans.append(cur_cam_trans)
            self.select_cam_rot.append(cur_cam_euler)
            self.select_cam_int.append(cur_fov / 180 * np.pi)

    def config_light(self):
        same_light = False

        self.select_light_path = []
        self.select_light_rotz = []
        self.select_light_rotx = []

        if same_light:  # for debug usage
            self.select_light_path = [opt.light_cfg] * self.num_view
            self.select_light_rotz = [0.] * self.num_view
            self.select_light_rotx = [-90] * self.num_view
        else:
            for i in range(self.num_view):
                light_type = self._sample_light_type()
                light_pth_list = os.listdir(os.path.join(HDRI_ROOT, light_type))
                light_pth = random.choice(light_pth_list)
                self.select_light_path.append(
                    os.path.join(HDRI_ROOT, light_type, light_pth)
                )
                self.select_light_rotz.append(
                    random.randint(0, 18 - 1) * 20
                )
                self.select_light_rotx.append(
                    -(60 + random.randint(0, 6) * 5)
                )


def create_image_node(nodes, image_path, color_space):
    assert color_space in ["Non-Color", "Linear", "sRGB"]
    image_node = nodes.new('ShaderNodeTexImage')
    image_node.image = bpy.data.images.load(image_path, check_existing=True)
    image_node.image.colorspace_settings.name = color_space
    return image_node


def set_material(export_dir, materials):
    base_color_path = os.path.join(export_dir, "diff_sr_4x.png")
    spec_color_path = os.path.join(export_dir, "spec_sr_4x.png")
    normal_path = os.path.join(export_dir, "normal_sr_4x.png")

    nodes = materials.nodes
    links = materials.links
    principled_bsdf = materials.get_the_one_node_with_type("BsdfPrincipled")

    base_color_node = create_image_node(nodes, base_color_path, "sRGB")
    spec_color_node = create_image_node(nodes, spec_color_path, "Non-Color")
    normal_node = create_image_node(nodes, normal_path, "Non-Color")

    links.new(base_color_node.outputs["Color"], principled_bsdf.inputs["Base Color"])
    links.new(spec_color_node.outputs["Color"], principled_bsdf.inputs["Specular IOR Level"])
    principled_bsdf.inputs["Roughness"].default_value = 0.45
    
    if opt.sss == 1:
        principled_bsdf.inputs["Subsurface Weight"].default_value = 1.0
        principled_bsdf.inputs["Subsurface Scale"].default_value = 0.0025
    
    # add normal map
    normal_map = nodes.new("ShaderNodeNormalMap")
    normal_map.space = "TANGENT"
    normal_map.inputs["Strength"].default_value = 1.
    links.new(normal_node.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], principled_bsdf.inputs["Normal"])


def set_geometry(obj_path, materials):
    obj = bproc.loader.load_obj(obj_path)
    face_obj = obj[0]
    face_obj.add_material(materials)
    face_obj.set_rotation_euler(np.array([np.pi, 0., 0.]))
    
    face_obj.blender_obj.name = "face_scan"


def create_scene(obj_path, export_dir):
    materials = bproc.material.create("avatar")
    set_geometry(obj_path, materials)
    set_material(export_dir, materials)


def set_world_background_hdr_img(
    path_to_hdr_file: str, 
    strength: float = 1.0,
    rot_z: float = 0.,
    rot_x: float = 0.
):
    """
    Sets the world background to the given hdr_file.

    :param path_to_hdr_file: Path to the .hdr file
    :param strength: The brightness of the background.
    """

    if not os.path.exists(path_to_hdr_file):
        raise FileNotFoundError(f"The given path does not exists: {path_to_hdr_file}")

    world = bpy.context.scene.world
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # add a texture node and load the image and link it
    texture_node = nodes.new(type="ShaderNodeTexEnvironment")
    texture_node.image = bpy.data.images.load(path_to_hdr_file, check_existing=True)

    # get the one background node of the world shader
    background_node = Utility.get_the_one_node_with_type(nodes, "Background")

    # link the new texture node to the background
    links.new(texture_node.outputs["Color"], background_node.inputs["Color"])

    # Set the brightness of the background
    background_node.inputs["Strength"].default_value = strength

    mapping_node = nodes.new(type="ShaderNodeMapping")
    links.new(mapping_node.outputs["Vector"], texture_node.inputs["Vector"])
    mapping_node.inputs["Rotation"].default_value[2] =  rot_z
    mapping_node.inputs["Rotation"].default_value[0] =  rot_x

    tex_coord_node = nodes.new(type="ShaderNodeTexCoord")
    links.new(tex_coord_node.outputs["Generated"], mapping_node.inputs["Vector"])


export_dir = opt.data_root
obj_path = os.path.join(opt.data_root, "final_hack_aligned.obj")
create_scene(obj_path, export_dir)

render_cfg = RenderConfig()

bproc.renderer.set_max_amount_of_samples(256)
bproc.renderer.set_light_bounces(
    diffuse_bounces=4,
    glossy_bounces=4,
    transmission_bounces=12,
    max_bounces=12,
)
bpy.ops.object.shade_smooth()

save_root = opt.save_root
img_save_root = os.path.join(save_root, "image")
diff_save_root = os.path.join(save_root, "diffuse")
os.makedirs(img_save_root, exist_ok=True)
os.makedirs(diff_save_root, exist_ok=True)


info = {
    "scene": {
        "obj_path": obj_path,
        "export_dir": export_dir,
    },
    "view": {}
}

for i in range(render_cfg.num_view):
    bproc.utility.reset_keyframes()

    bproc.camera.set_intrinsics_from_blender_params(
        lens=render_cfg.select_cam_int[i], 
        lens_unit="FOV", 
        image_height=1024, 
        image_width=1024, 
    )

    cam_pose = bproc.math.build_transformation_mat(
        render_cfg.select_cam_trans[i], 
        render_cfg.select_cam_rot[i],
    )
    bproc.camera.add_camera_pose(cam_pose)
    bproc.renderer.enable_diffuse_color_output()
    # bproc.renderer.enable_normals_output()
    bproc.renderer.set_output_format()

    set_world_background_hdr_img(
        path_to_hdr_file=render_cfg.select_light_path[i],
        rot_z=render_cfg.select_light_rotz[i]/180*np.pi,
        rot_x=render_cfg.select_light_rotx[i]/180*np.pi,
    )

    bpy.data.objects['face_scan'].visible_shadow = True
    data = bproc.renderer.render()
    cv2.imwrite(
        os.path.join(img_save_root, "%05d.png" % i), data["colors"][0][..., [2, 1, 0]]
    )
    cv2.imwrite(
        os.path.join(diff_save_root, "%05d.png" % i), data["diffuse"][0][..., [2, 1, 0]]
    )

    info["view"][i] = {
        "cam": {
            "fov": render_cfg.select_cam_int[i],
            "rot": render_cfg.select_cam_rot_mat[i],
            "trans": render_cfg.select_cam_trans[i],
        },
        "light": {
            "path": render_cfg.select_light_path[i],
            "rotz": render_cfg.select_light_rotz[i],
            "rotx": render_cfg.select_light_rotx[i],
        },
    }


np.save(os.path.join(save_root, "info.npy"), info)
