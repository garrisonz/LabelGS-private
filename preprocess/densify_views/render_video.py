#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.eval_tools import interpolate_views

def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, args, texts, label_id):

    if views is None or len(views) == 0:
        print("No views to render", name)
        return
    feature_tag = ""
    text_str = texts + "_" + str(label_id)+"_"
    render_path = os.path.join(model_path, name, "{}{}{}".format(text_str, iteration, feature_tag), "renders")
    gts_path = os.path.join(model_path, name, "{}{}".format(text_str, iteration, feature_tag), "gt")
    render_npy_path = os.path.join(model_path, name, "{}{}{}".format(text_str, iteration, feature_tag), "renders_npy")
    gts_npy_path = os.path.join(model_path, name, "{}{}{}".format(text_str, iteration, feature_tag), "gt_npy")
    mapping_path = os.path.join(model_path, name, "{}{}{}".format(text_str, iteration, feature_tag))

    makedirs(render_npy_path, exist_ok=True)
    makedirs(gts_npy_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # interpolation of adjacent views
    #views = scene.getTrainCameras()
    print("views:", len(views))
    for view in views:
        print(view.image_name, end=" ")
    # interpolate adjacent views for rendering
    views = interpolate_views(views)
    print("interpolated views:", len(views))

    
    rendering_gt_mapping = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        output = render(view, gaussians, pipeline, background, args, label_id=label_id)

        rendering = output["render"]

        # if view is a object of Camera class
        from scene.cameras import Camera
        if isinstance(view, Camera):
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            rendering_gt_mapping[view.image_name] = '{0:05d}'.format(idx)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

    # save rendering_gt_mapping as a json file
    import json
    with open(os.path.join(mapping_path, 'rendering_gt_mapping.json'), 'w') as f:
        json.dump(rendering_gt_mapping, f)
    
    print("rendering done", render_path)
    return render_path


def get_label_id(text_str, scene_path):
    import open_clip
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    text = tokenizer([text_str])

    model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained="laion2b_s34b_b88k")
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    text_embed = model.encode_text(text)
    text_embed = text_embed.to("cuda").float()
    print("text_embed:", text_embed.shape)

    image_embeds_path = f"{scene_path}/clip_embeds/image_embeds.npy"
    image_embeds = np.load(image_embeds_path)

    mask_ids_path = f"{scene_path}/clip_embeds/mask_ids.npy"
    mask_ids = np.load(mask_ids_path)

    image_embeds = torch.tensor(image_embeds, dtype=torch.float32, device="cuda")
    print("image_embeds:", image_embeds.shape)
    
    with torch.no_grad():
        text_embed /= text_embed.norm(dim=-1, keepdim=True)
        image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * text_embed @ image_embeds.T).softmax(dim=-1)
        print("mask_ids:", mask_ids)
        print("Label probs:", text_probs)
        match_idx = torch.argmax(text_probs)
        print("most similar object:", match_idx, mask_ids[match_idx])

    return mask_ids[match_idx], text_probs[0][match_idx]


def images_to_video2(image_folder, output_path, fps=10):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()

    print(image_folder)

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)
    video.release()

def filter_views(scene_path, full_views):
    view_all_list_file = scene_path + "/view_all_list.txt"
    if os.path.exists(view_all_list_file):
        views = []
        with open(view_all_list_file, "r") as f:
            range_list = f.readlines()
            range_list = [range.strip() for range in range_list]
            range_list = [range for range in range_list if range.startswith("#") is False]
            # range is in format start_num-end_num
            # check if view.image_name is in any of the range
            valid_list = []
            for range_str in range_list:
                start, end = range_str.split("-")
                for i in range(int(start), int(end)+1):
                    valid_list.append(f"frame_{i:05d}")
            print(valid_list)
            views = [view for view in full_views if view.image_name in valid_list]
        return views
    else:
        return full_views

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):

    # encode text by openclipnetwork
    text_str = args.text
    if text_str == "":
        label_id = None
        print("label_id:", label_id)
    else:
        scene_path = dataset.source_path
        label_id, score = get_label_id(text_str, scene_path)
        print("label_id:", label_id, "score:", score)
    
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)

        checkpoint = os.path.join(args.model_path, f'chkpnt{args.loaded_iter}.pth')
        print(f"loading checkpoint: {checkpoint}")
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            views = scene.getTrainCameras()
            print("views:", len(views))

            render_path = render_set(dataset.model_path, dataset.source_path, "train", args.loaded_iter, views, gaussians, pipeline, background, args, texts=text_str, label_id=label_id)
            #image_to_video(render_path, f"{dataset.model_path}/train_{text_str}_{label_id}_{args.loaded_iter}.avi")
            images_to_video2(render_path, f"{dataset.model_path}/train_{text_str}_{label_id}_{args.loaded_iter}.mp4", 10)

        if not skip_test:
             render_set(dataset.model_path, dataset.source_path, "test", args.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args, texts=text_str, label_id=label_id)

if __name__ == "__main__":
    # Set up command line argument parser
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--loaded_iter", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--text', type=str, default='')

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)