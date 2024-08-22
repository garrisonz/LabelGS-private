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

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def render_views(views, view_type, gaussians, pipeline, background, args, render_path):
    for view in views:
        image_name = view.image_name

        render_pkg = render(view, gaussians, pipeline, background, args)
        render_img = render_pkg["render"].cpu()
        alpha_id_map = render_pkg["alpha_id_map"].cpu()
        torchvision.utils.save_image(render_img,  render_path+f"/{view_type}_{image_name}.png")

        gaussian_label = gaussians.label
        alpha_id_map = alpha_id_map.type(torch.long)
        label_map = gaussian_label[alpha_id_map]

        save_label_map = label_map.clone().type(torch.float32) / label_map.max()
        torchvision.utils.save_image(save_label_map,  render_path+f"/{view_type}_{image_name}_label_map.png")

        # save for debug
        uni_mask_map = view.mask_list[0]
        label_ids = torch.unique( uni_mask_map).type(torch.long)
        for label_id in label_ids:
            print(image_name, ",label_id:", label_id)
            label_map_label = label_map == label_id
            save_label_map_label = label_map_label.clone().type(torch.float32) * 255
            torchvision.utils.save_image(save_label_map_label,  render_path+f"/{view_type}_{image_name}_label_map_{label_id}.png")

            uni_label_map_label = uni_mask_map == label_id
            uni_label_map_label = uni_label_map_label.type(torch.float32) * 255
            torchvision.utils.save_image(uni_label_map_label,  render_path+f"/{view_type}_{image_name}_label_map_{label_id}_gt.png")

            if label_id == 0 or label_id == -1:
                continue
            query_id = [label_id]
            query_id = torch.tensor(query_id, dtype=torch.int32, device="cuda")
            render_pkg = render(view, gaussians, pipeline, background, args, label_id=query_id)
            torchvision.utils.save_image(render_pkg["render"],  render_path+f"/{view_type}_{image_name}_label_{label_id}.png")


def render_process(dataset : ModelParams, pipeline : PipelineParams, args):
    dataset.eval = True

    scene_name = args.model_path.split("/")[-1][5:-9]
    print(scene_name)
    dataset_name = args.model_path.split("/")[-2]
    print(dataset_name)

    render_path = args.model_path + f"/discret{args.loaded_iter}"
    makedirs(render_path, exist_ok=True)

    with torch.no_grad():
        # read gaussian model
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)
        checkpoint = os.path.join(args.model_path, f'chkpnt{args.loaded_iter}.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        train_cameras = scene.getTrainCameras()
        # random sample 3 views from train_cameras
        train_cameras = np.random.choice(train_cameras, 3, replace=False)
        render_views(train_cameras, "train", gaussians, pipeline, background, args, render_path)

        test_cameras = scene.getTestCameras()
        render_views(test_cameras, "test", gaussians, pipeline, background, args, render_path)



        

# task: extract 3d object in gaussian model from prompt, and evaluate on eval_set
# pipeline: 
# 1. read gaussian model
# 2. extract 3d object in gaussian model from prompt
# 3. evaluate on eval_set

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--loaded_iter", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_process(model.extract(args), pipeline.extract(args), args)


