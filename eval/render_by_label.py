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
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.eval_tools import interpolate_views

def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, args, label_id):
    feature_tag = ""
    render_path = os.path.join(model_path, name, "{}_{}{}".format(label_id, iteration, feature_tag), "renders")
    gts_path = os.path.join(model_path, name, "{}_{}{}".format(label_id, iteration, feature_tag), "gt")
    render_npy_path = os.path.join(model_path, name, "{}_{}{}".format(label_id, iteration, feature_tag), "renders_npy")
    gts_npy_path = os.path.join(model_path, name, "{}_{}{}".format(label_id, iteration, feature_tag), "gt_npy")

    makedirs(render_npy_path, exist_ok=True)
    makedirs(gts_npy_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    views = interpolate_views(views)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if label_id == -1:
            output = render(view, gaussians, pipeline, background, args)
        else:
            output = render(view, gaussians, pipeline, background, args, label_id=torch.tensor(label_id))

        rendering = output["render"]

        gt = view.original_image[0:3, :, :]

        np.save(os.path.join(render_npy_path, '{0:05d}'.format(idx) + ".npy"),rendering.permute(1,2,0).cpu().numpy())
        np.save(os.path.join(gts_npy_path, '{0:05d}'.format(idx) + ".npy"),gt.permute(1,2,0).cpu().numpy())
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
               
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False, load_iteration=args.loaded_iter)
        checkpoint = os.path.join(args.model_path, f'chkpnt{args.loaded_iter}.pth')
        print("checkpoint:", checkpoint)
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, dataset.source_path, "train", args.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args, label_id=args.label_id)

        if not skip_test:
             render_set(dataset.model_path, dataset.source_path, "test", args.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args, label_id=args.label_id)

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
    parser.add_argument("--label_id", default=-1, type=int)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)