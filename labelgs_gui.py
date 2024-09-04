# Borrowed from OmniSeg3D-GS (https://github.com/OceanYing/OmniSeg3D-GS)
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
import torch
import datetime
from scene import Scene
import os
from gaussian_renderer import render
from argparse import ArgumentParser
import numpy as np
from PIL import Image
import cv2

from scene import GaussianModel
import dearpygui.dearpygui as dpg
import math
from scene.cameras import Camera, CameraBase
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

from gui.OrbitCamera import OrbitCamera
from utils.eval_tools import interpolate_views

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

        self.gaussians = self.engine['scene']

        print("loading model file done.")

        self.mode = "image"

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
        self.click_time = ""
        self.click_multiview_num = 0
        self.new_click_xy = []
        self.clear_edit = False
        self.next_view = False
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

        self.cameras = get_camera_pose(opt.SOURCE_PATH, self.height, self.width)
        

        self.smooth_cameras = interpolate_views(self.cameras)
        self.playing = False
        self.reset_playing = False
        self.reset_fovy1 = False

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
        def playing():
            self.playing = 1 - self.playing
            print("playing:", self.playing)
            #if self.playing == False:
            #    self.camera.update(self.smooth_cameras[self.frame_id])
        def reset_playing():
            self.reset_playing = True
        def reset_fovy1():
            self.reset_fovy1 = True

        # control window
        with dpg.window(label="Control", tag="_control_window", width=300, height=550, pos=[self.window_width+10, 0]):

            dpg.add_text("\nSegment option: ", tag="seg")
            dpg.add_checkbox(label="clickmode", callback=clickmode_callback, user_data="Some Data", default_value=True)
            
            dpg.add_text("\n")
            #dpg.add_button(label="next view", callback=next_view, user_data="Some Data")
            dpg.add_button(label="segment3d", callback=callback_segment3d, user_data="Some Data")
            dpg.add_button(label="clear", callback=clear_edit, user_data="Some Data")
            dpg.add_button(label="saves as", callback=callback_save, user_data="Some Data")
            dpg.add_input_text(label="", default_value="current", tag="save_name")
            dpg.add_button(label="gaussain mask saves as", callback=callback_save_gaussain_mask, user_data="Some Data")
            dpg.add_input_text(label="", default_value="gaussian_mask", tag="gaussian_mask_name")
            dpg.add_text("selected gaussian:", tag="pos_item")
            dpg.add_checkbox(label="playing", callback=playing, user_data="Some Data", default_value=False)
            dpg.add_button(label="reset_playing", callback=reset_playing, user_data="Some Data")
            dpg.add_text("frame id:", tag="frame_id")
            dpg.add_slider_float(label="Scale", default_value=1.0, min_value=0.001, max_value=1.0, tag="_Scale")
            dpg.add_button(label="training fovy", callback=reset_fovy1, user_data="Some Data")
            dpg.add_slider_float(label="fovy", default_value=60.0, min_value=1.0, max_value=60.0, tag="_fovy")
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
                self.new_click_xy = np.array(xy)
                self.new_click = True
                now = datetime.datetime.now()
                self.click_time = now.strftime("%d_%H%M%S")
                self.click_multiview_num = 0


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
                if self.playing:
                    cam = self.smooth_cameras[self.frame_id]
                    self.camera.update(cam)
                cam = self.construct_camera()
                self.fetch_data(cam)
            dpg.render_dearpygui_frame()

    def construct_camera(
        self,
    ) -> CameraBase:
        if self.camera.rot_mode == 1:
            pose = self.camera.pose_movecenter
        elif self.camera.rot_mode == 0:
            pose = self.camera.pose_objcenter

        R = pose[:3, :3]
        t = pose[:3, 3]

        fovy = self.camera.fovy

        fy = fov2focal(fovy, self.height)
        fovx = focal2fov(fy, self.width)

        cam = CameraBase(
            colmap_id=0,
            R=R,
            T=t,
            FoVx=fovx,
            FoVy=fovy,
            image_name=None,
            image_height=self.height,
            image_width=self.width,
            uid=0,
        )
        return cam
    
    @torch.no_grad()
    def fetch_data(self, view_camera):

        if self.reset_fovy1:
            dpg.set_value("_fovy", self.cameras[0].FoVy * 180 / math.pi)

        gaussians = self.gaussians
        scale = dpg.get_value('_Scale')
        fovy = dpg.get_value('_fovy')
        self.camera.update_fovy_degrees(fovy)

        scene_outputs = render(view_camera, self.engine['scene'], self.opt, self.bg_color, scaling_modifier=scale)
        img = scene_outputs["render"].permute(1, 2, 0)

        if self.new_click:
            img_path = f"./{self.save_folder}/{self.click_time}_all_gua_render.png"
            save_image(img, img_path)

        if self.clear_edit:
            self.new_click_xy = []
            self.clear_edit = False
            
            self.label_list = []
            self.segment3d_flag = False

        if self.segment3d_flag:
            if len(self.label_list) > 0:
                labels = torch.tensor(self.label_list, device="cuda")
                scene_outputs = render(view_camera, self.engine['scene'], self.opt, self.bg_color, label_id=labels, scaling_modifier=scale)
                img = scene_outputs["render"].permute(1, 2, 0)
        else:
            if self.new_click:
                mask = self.point_to_mask(self.new_click_xy, img)
                mask_save = mask.repeat(3, 1, 1).permute(1, 2, 0)

                save_path = f"./{self.save_folder}/{self.click_time}_2d_mask.png"
                save_image(mask_save, save_path)

                alpha_id_map = scene_outputs["alpha_id_map"].cpu()

                alpha_id_map = alpha_id_map.type(torch.long)

                label_map = gaussians.label[alpha_id_map].cpu().numpy()
                label_map_cnt = np.unique(label_map, return_counts=True)
                label_map_cnt = dict(zip(label_map_cnt[0], label_map_cnt[1]))

                mask_label_cnt = np.unique(label_map * mask.cpu().numpy(), return_counts=True)
                mask_label_cnt = dict(zip(mask_label_cnt[0], mask_label_cnt[1]))
                mask_label_cnt.pop(0, None)
                mask_label_cnt.pop(-1, None)
                for k, v in mask_label_cnt.items():
                    if v > label_map_cnt[k] * 0.5:
                        self.label_list.append(k)
                self.label_list = list(set(self.label_list))

            if len(self.label_list) > 0:
                labels = torch.tensor(self.label_list, device="cuda")
                scene_outputs = render(view_camera, self.engine['scene'], self.opt, self.bg_color, label_id=labels, scaling_modifier=scale)
                mask_pred = scene_outputs["render"] > 0.01
                mask = mask_pred.any(dim=0)
                mask = mask.repeat(3, 1, 1).permute(1, 2, 0)
                alpha = (mask == 0).float() * 0.5 + 0.5
                img = img * alpha + mask * (1 - alpha)

                if self.new_click:
                    save_path = f"./{self.save_folder}/{self.click_time}_3d_mask.png"
                    save_image(mask, save_path)

                    save_img = scene_outputs["render"].permute(1, 2, 0)
                    save_path = f"./{self.save_folder}/{self.click_time}_3D_seg.png"
                    save_image(save_img, save_path)

        if self.new_click:
            save_img = img.clone()
            img_path = f"./{self.save_folder}/{self.click_time}_display.png"
            xy = [int(self.new_click_xy[0]), int(self.new_click_xy[1])]
            save_img[xy[1]-5:xy[1]+5, xy[0]-5:xy[0]+5] = torch.tensor([1, 0, 0], device="cuda")
            save_image(save_img, img_path)

        if self.save_gaussian_mask:
            self.save_gaussian_mask_func()

        if self.save_flag:
            save_path = f"./{self.save_folder}/{self.click_time}_{dpg.get_value('save_name')}_{self.click_multiview_num}.png"
            print("save:", save_path)
            save_image(img, save_path)
            self.click_multiview_num += 1

        if self.playing:
            if self.frame_id < len(self.smooth_cameras) + 1:
                self.frame_id += 1
            else:
                self.frame_id = 0
        if self.reset_playing:
            self.frame_id = 0

        self.update_status(img)

    def update_status(self, img):
        gaussians = self.gaussians
        gau_mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        for label in self.label_list:
            gau_mask = gau_mask | (gaussians.label == label)
        selected_gau_num = gau_mask.sum().item()
        dpg.set_value("pos_item", f"selected gaussians: {selected_gau_num} / {gaussians.get_xyz.shape[0]}")
        dpg.set_value("frame_id", f"frame id: {self.frame_id} / {len(self.smooth_cameras)}")

        self.render_buffer = img.clone().cpu().numpy().reshape(-1)
        dpg.set_value("_texture", self.render_buffer)

        self.new_click = False
        self.save_flag = False
        self.save_gaussian_mask = False
        self.reset_playing = False
        self.reset_fovy1 = False

    def save_gaussian_mask_func(self):
        gaussians = self.gaussians
        gau_mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        for label in self.label_list:
            gau_mask = gau_mask | (gaussians.label == label)
        mask_path = f"./{self.save_folder}/{dpg.get_value('gaussian_mask_name')}.npy"
        torch.save(gau_mask, mask_path)
        print("Saved to ", mask_path)


    @torch.no_grad()
    def point_to_mask(self, xy, image):
        image = (image.cpu().numpy()*255).astype(np.uint8)

        sam = self.predictor
        sam.set_image(image)

        input_point = np.array([xy]).astype(np.int32)
        input_label = np.array([1])

        masks, scores, logits = sam.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        return torch.tensor(masks[np.argmax(scores)], device="cuda")

def save_image(img, path):
    img = img.cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)

def get_camera_pose(scene_path, image_height, image_width):
    cameras_extrinsic_file = os.path.join(scene_path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(scene_path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    cameras = []
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        R = np.transpose(qvec2rotmat(extr.qvec))
        t = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)

        cam = CameraBase(
            colmap_id=0,
            R=R,
            T=t,
            FoVx=FovX,
            FoVy=FovY,
            image_name=None,
            image_height=image_height,
            image_width=image_width,
            uid=0,
        )

        cameras.append(cam)

    return cameras


if __name__ == "__main__":
    parser = ArgumentParser(description="GUI option")

    parser.add_argument('-m', '--model_path', type=str, default="./output/figurines")
    parser.add_argument('--scene_iteration', type=int, default=15000)
    parser.add_argument('-s', '--source_path', type=str, default=None)

    args = parser.parse_args()

    from gui.CONFIG import CONFIG
    opt = CONFIG()

    opt.MODEL_PATH = args.model_path
    opt.SCENE_GAUSSIAN_ITERATION = args.scene_iteration
    opt.SOURCE_PATH = args.source_path

    opt.SCENE_PCD_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.SCENE_GAUSSIAN_ITERATION)}/point_cloud.ply')

    gs_model = GaussianModel(opt.sh_degree)
    gui = GaussianSplattingGUI(opt, gs_model)

    gui.render()