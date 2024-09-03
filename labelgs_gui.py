# Borrowed from OmniSeg3D-GS (https://github.com/OceanYing/OmniSeg3D-GS)
import torch
from scene import Scene
import os
from tqdm import tqdm
from gaussian_renderer import render
import torchvision
from argparse import ArgumentParser
# from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import cv2

# from scene.gaussian_model import GaussianModel
from scene import GaussianModel
import dearpygui.dearpygui as dpg
import math
from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

from scipy.spatial.transform import Rotation as R

class CONFIG:
    r = 2 
    window_width = int(1440/r)
    window_height = int(1080/r)

    width = int(1440/r)
    height = int(1080/r)

    radius = 2

    debug = False
    dt_gamma = 0.2

    # gaussian model
    sh_degree = 3

    convert_SHs_python = False
    compute_cov3D_python = False

    white_background = False

    MODEL_PATH = './output/'

    SCENE_GAUSSIAN_ITERATION = 15000

    SCENE_PCD_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(SCENE_GAUSSIAN_ITERATION)}/point_cloud.ply')


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_quat(
            [0, 0, 0, 1]
        )  # init camera matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!
        self.right = np.array([1, 0, 0], dtype=np.float32)  # need to be normalized!
        self.fovy = fovy
        self.translate = np.array([0, 0, self.radius])
        self.scale_f = 1.0


        self.rot_mode = 1   # rotation mode (1: self.pose_movecenter (movable rotation center), 0: self.pose_objcenter (fixed scene center))
        # self.rot_mode = 0


    @property
    def pose_movecenter(self):
        # --- first move camera to radius : in world coordinate--- #
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        
        # --- rotate: Rc --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # --- translate: tc --- #
        res[:3, 3] -= self.center
        
        # --- Convention Transform --- #
        # now we have got matrix res=c2w=[Rc|tc], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]
        res[:3, 3] = -rot[:3, :3].transpose() @ res[:3, 3]
        
        return res
    
    @property
    def pose_objcenter(self):
        res = np.eye(4, dtype=np.float32)
        
        # --- rotate: Rw --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # --- translate: tw --- #
        res[2, 3] += self.radius    # camera coordinate z-axis
        res[:3, 3] -= self.center   # camera coordinate x,y-axis
        
        # --- Convention Transform --- #
        # now we have got matrix res=w2c=[Rw|tw], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]=[Rw.T|tw]
        res[:3, :3] = rot[:3, :3].transpose()
        
        return res

    def orbit(self, dx, dy):
        if self.rot_mode == 1:    # rotate the camera axis, in world coordinate system
            up = self.rot.as_matrix()[:3, 1]
            side = self.rot.as_matrix()[:3, 0]
        elif self.rot_mode == 0:    # rotate in camera coordinate system
            up = -self.up
            side = -self.right
        rotvec_x = up * np.radians(0.01 * dx)
        rotvec_y = side * np.radians(0.01 * dy)

        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        # self.radius *= 1.1 ** (-delta)    # non-linear version
        self.radius -= 0.1 * delta      # linear version

    def pan(self, dx, dy, dz=0):
        
        if self.rot_mode == 1:
            # pan in camera coordinate system: project from [Coord_c] to [Coord_w]
            self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])
        elif self.rot_mode == 0:
            # pan in world coordinate system: at [Coord_w]
            self.center += 0.0005 * np.array([-dx, dy, dz])


class GaussianSplattingGUI:
    def __init__(self, opt, gaussian_model:GaussianModel) -> None:
        self.opt = opt

        self.width = opt.width
        self.height = opt.height
        self.window_width = opt.window_width
        self.window_height = opt.window_height
        self.camera = OrbitCamera(opt.width, opt.height, r=opt.radius)

        bg_color = [1, 1, 1] if opt.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.bg_color = background
        self.render_buffer = np.zeros((self.width, self.height, 3), dtype=np.float32)
        self.update_camera = True
        self.dynamic_resolution = True
        self.debug = opt.debug
        self.engine = {'scene': gaussian_model,}

        self.load_model = False
        print("loading model file:", self.opt.SCENE_PCD_PATH)
        self.engine['scene'].load_ply(self.opt.SCENE_PCD_PATH)
        self.load_model = True

        print("loading model file done.")

        self.mode = "image"  # choose from ['image', 'depth']

        dpg.create_context()
        self.register_dpg()

        self.frame_id = 0

        # --- for better operation --- #
        self.moving = False
        self.moving_middle = False
        self.mouse_pos = (0, 0)

        # --- for interactive segmentation --- #
        self.img_mode = 0
        self.clickmode_button = True
        self.new_click = False
        self.new_click_xy = []
        self.clear_edit = False                 # clear all the click prompts
        self.segment3d_flag = False

        self.render_mode_rgb = False

        self.save_flag = False
        self.save_gaussian_mask = False
        self.save_folder = opt.MODEL_PATH + "/seg_images"
        os.makedirs(self.save_folder, exist_ok=True)

        from segment_anything import SamPredictor, sam_model_registry
        sam_checkpoint = "ckpts/sam/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.predictor = SamPredictor(sam)

        self.mask = None
        self.label_list = []


    def __del__(self):
        dpg.destroy_context()

    def prepare_buffer(self, outputs):
        if self.model == "images":
            return outputs["render"]
        else:
            return np.expand_dims(outputs["depth"], -1).repeat(3, -1)

    def register_dpg(self):
        
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.width, self.height, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window
        with dpg.window(tag="_primary_window", width=self.window_width+300, height=self.window_height):
            dpg.add_image("_texture")   # add the texture

        dpg.set_primary_window("_primary_window", True)

        # --- interactive mode switch --- #
        def clickmode_callback(sender):
            self.clickmode_button = 1 - self.clickmode_button
        def clear_edit():
            self.clear_edit = True
        def callback_segment3d():
            self.segment3d_flag = True
        def callback_save():
            self.save_flag = True
        def callback_save_gaussain_mask():
            self.save_gaussian_mask = True

        def render_mode_rgb_callback(sender):
            self.render_mode_rgb = not self.render_mode_rgb
        # control window
        with dpg.window(label="Control", tag="_control_window", width=300, height=550, pos=[self.window_width+10, 0]):

            dpg.add_text("\nSegment option: ", tag="seg")
            dpg.add_checkbox(label="clickmode", callback=clickmode_callback, user_data="Some Data", default_value=True)
            
            dpg.add_text("\n")
            dpg.add_button(label="segment3d", callback=callback_segment3d, user_data="Some Data")
            dpg.add_button(label="clear", callback=clear_edit, user_data="Some Data")
            dpg.add_button(label="rendering saves as", callback=callback_save, user_data="Some Data")
            dpg.add_input_text(label="", default_value="image", tag="save_name")
            dpg.add_button(label="gaussain mask saves as", callback=callback_save_gaussain_mask, user_data="Some Data")
            dpg.add_input_text(label="", default_value="gaussian_mask", tag="gaussian_mask_name")
            dpg.add_text("selected gaussian:", tag="pos_item")
            dpg.add_text("\n")

        if self.debug:
            with dpg.collapsing_header(label="Debug"):
                dpg.add_separator()
                dpg.add_text("Camera Pose:")
                dpg.add_text(str(self.camera.pose), tag="_log_pose")


        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            delta = app_data
            self.camera.scale(delta)
            self.update_camera = True
            if self.debug:
                dpg.set_value("_log_pose", str(self.camera.pose))
        
        def toggle_moving_left():
            self.moving = not self.moving

        def toggle_moving_middle():
            self.moving_middle = not self.moving_middle

        def move_handler(sender, pos, user):
            if self.moving and dpg.is_item_focused("_primary_window"):
                dx = self.mouse_pos[0] - pos[0]
                dy = self.mouse_pos[1] - pos[1]
                if dx != 0.0 or dy != 0.0:
                    self.camera.orbit(-dx*30, dy*30)
                    self.update_camera = True

            if self.moving_middle and dpg.is_item_focused("_primary_window"):
                dx = self.mouse_pos[0] - pos[0]
                dy = self.mouse_pos[1] - pos[1]
                if dx != 0.0 or dy != 0.0:
                    self.camera.pan(-dx*20, dy*20)
                    self.update_camera = True
            
            self.mouse_pos = pos

        def change_pos(sender, app_data):
            xy = dpg.get_mouse_pos(local=False)
            if self.clickmode_button and app_data == 1:     # in the click mode and right click
                #print(xy)
                self.new_click_xy = np.array(xy)
                self.new_click = True


        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Left, callback=lambda:toggle_moving_left())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Left, callback=lambda:toggle_moving_left())
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Middle, callback=lambda:toggle_moving_middle())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Middle, callback=lambda:toggle_moving_middle())
            dpg.add_mouse_move_handler(callback=lambda s, a, u:move_handler(s, a, u))
            
            dpg.add_mouse_click_handler(callback=change_pos)
            
        dpg.create_viewport(title="Gaussian-Splatting-Viewer", width=self.window_width+320, height=self.window_height, resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        dpg.show_viewport()


    def render(self):
        while dpg.is_dearpygui_running():
            if self.load_model:
                cam = self.construct_camera()
                self.fetch_data(cam)
            dpg.render_dearpygui_frame()


    def construct_camera(
        self,
    ) -> Camera:
        if self.camera.rot_mode == 1:
            pose = self.camera.pose_movecenter
        elif self.camera.rot_mode == 0:
            pose = self.camera.pose_objcenter

        R = pose[:3, :3]
        t = pose[:3, 3]

        ss = math.pi / 180.0
        fovy = self.camera.fovy * ss

        fy = fov2focal(fovy, self.height)
        fovx = focal2fov(fy, self.width)

        mask_list = []
        sdf_list = []
        occlude_mapping = []

        cam = Camera(
            colmap_id=0,
            R=R,
            T=t,
            FoVx=fovx,
            FoVy=fovy,
            image=torch.zeros([3, self.height, self.width]),
            gt_alpha_mask=None,
            image_name=None,
            uid=0,
            mask_list=mask_list,
            sdf_list=sdf_list,
            occlude_mapping=occlude_mapping,
        )
        return cam
    
    @torch.no_grad()
    def fetch_data(self, view_camera):
        gaussians = self.engine["scene"]

        if self.segment3d_flag and (self.label_list) is not None:
            labels = torch.tensor(self.label_list, device="cuda")
            scene_outputs = render(view_camera, self.engine['scene'], self.opt, self.bg_color, label_id=labels)
        else:
            scene_outputs = render(view_camera, self.engine['scene'], self.opt, self.bg_color)
        img = scene_outputs["render"].permute(1, 2, 0)

        if self.clear_edit:
            self.new_click_xy = []
            self.clear_edit = False
            
            self.mask = None
            self.label_list = []
            self.segment3d_flag = False

        if self.new_click:
            print("selected:", self.new_click_xy)
            mask = self.point_to_mask(self.new_click_xy, img)
            self.mask = mask

            self.new_click = False

            alpha_id_map = scene_outputs["alpha_id_map"].cpu()

            alpha_id_map = alpha_id_map.type(torch.long)
            gaussians = self.engine['scene']

            label_map = gaussians.label[alpha_id_map].cpu().numpy()
            label_map_cnt = np.unique(label_map, return_counts=True)
            label_map_cnt = dict(zip(label_map_cnt[0], label_map_cnt[1]))

            mask_label_cnt = np.unique(label_map * self.mask.cpu().numpy(), return_counts=True)
            mask_label_cnt = dict(zip(mask_label_cnt[0], mask_label_cnt[1]))
            mask_label_cnt.pop(0, None)
            mask_label_cnt.pop(-1, None)
            for k, v in mask_label_cnt.items():
                if v > label_map_cnt[k] * 0.5:
                    self.label_list.append(k)
            self.label_list = list(set(self.label_list))
            print("label_list:", self.label_list)

        if not self.segment3d_flag and len(self.label_list) > 0:
            labels = torch.tensor(self.label_list, device="cuda")
            scene_outputs = render(view_camera, self.engine['scene'], self.opt, self.bg_color, label_id=labels)
            mask_pred = scene_outputs["render"] > 0.01
            mask = mask_pred.any(dim=0)
            mask = mask.repeat(3, 1, 1).permute(1, 2, 0)
            alpha = (mask == 0).float() * 0.5 + 0.5
            img = img * alpha + mask * (1 - alpha)


        gau_mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        for label in self.label_list:
            gau_mask = gau_mask | (gaussians.label == label)
        selected_gau_num = gau_mask.sum().item()

        dpg.set_value("pos_item", f"selected gaussians: {selected_gau_num} / {gaussians.get_xyz.shape[0]}")

        self.render_buffer = img.clone().cpu().numpy().reshape(-1)

        dpg.set_value("_texture", self.render_buffer)

        if self.save_flag:
            self.save_flag = False
            # save rendered image
            img_path = f"./{self.save_folder}/{dpg.get_value('save_name')}.png"
            img_save = img.cpu().numpy()
            img_save = (img_save * 255).astype(np.uint8)
            img_save = cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, img_save)
            print("Saved to ", img_path)

        if self.save_gaussian_mask:
            self.save_gaussian_mask = False
            gau_mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
            for label in self.label_list:
                gau_mask = gau_mask | (gaussians.label == label)
            mask_path = f"./{self.save_folder}/{dpg.get_value('gaussian_mask_name')}.npy"
            torch.save(gau_mask, mask_path)
            print("Saved to ", mask_path)



    # --- point to mask --- #
    def point_to_mask(self, xy, image):
        # xy: [x, y]
        # image: [3, H, W]
        # self.predictor is sam model
        # get the mask from the sam model by clicking the point on the image

        # save image as RGB image
        img_path = f"{self.save_folder}/image_t1.png"
        img_save = image.cpu().numpy()
        img_save = (img_save * 255).astype(np.uint8)
        img_save = cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, img_save)

        # convert image to numpy ndarray, uint8
        image = (image.cpu().numpy()*255).astype(np.uint8)

        print("image:", image.shape, image.dtype)
        sam = self.predictor
        sam.set_image(image)

        input_point = np.array([xy]).astype(np.int32)
        input_label = np.array([1])
        print("input_points:", input_point)

        masks, scores, logits = sam.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        return torch.tensor(masks[np.argmax(scores)], device="cuda")

if __name__ == "__main__":
    parser = ArgumentParser(description="GUI option")

    parser.add_argument('-m', '--model_path', type=str, default="./output/figurines")
    parser.add_argument('-s', '--scene_iteration', type=int, default=15000)

    args = parser.parse_args()

    opt = CONFIG()

    opt.MODEL_PATH = args.model_path
    opt.SCENE_GAUSSIAN_ITERATION = args.scene_iteration

    opt.SCENE_PCD_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.SCENE_GAUSSIAN_ITERATION)}/point_cloud.ply')

    gs_model = GaussianModel(opt.sh_degree)
    gui = GaussianSplattingGUI(opt, gs_model)

    gui.render()