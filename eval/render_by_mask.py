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

def get_prompt_and_eval(seg_path):

    # each file in seg_path is a segmentation mask, with label on file name

    anno_all = {}
    frame_list = []
    for seg_frame in os.listdir(seg_path):
        if not os.path.isdir(os.path.join(seg_path, seg_frame)):
            continue
        frame_list.append(seg_frame)
    frame_list.sort()

    for seg_frame in frame_list:
        sef_frame_dir = os.path.join(seg_path, seg_frame)

        anno = {}
        for seg_name in os.listdir(sef_frame_dir):
            seg_file = os.path.join(sef_frame_dir, seg_name)
            label = seg_name.split(".")[0]
            #print(f"{seg_file}")

            # read the segmentation mask, paired with label, and stored in anno
            import cv2
            seg = cv2.imread(seg_file, cv2.IMREAD_GRAYSCALE)
            seg = seg > 128
            anno[label] = seg
        anno_all[seg_frame] = anno
    print(anno_all.keys())
    prompt = {}
    first = list(anno_all.keys())[0]
    prompt[first]= anno_all[first]
    eval_set = {k: v for k, v in anno_all.items() if k != first}
    return prompt, eval_set


def render_process(dataset : ModelParams, pipeline : PipelineParams, args):
    dataset.eval = True

    scene_name = args.model_path.split("/")[-1][5:-9]
    print(scene_name)
    dataset_name = args.model_path.split("/")[-2]
    print(dataset_name)
    seg_path = f"/home/zhangyupeng/w/3drecon/LabelGS/dataset/{dataset_name}/{scene_name}/mask_prompt"

    prompt, eval_set = get_prompt_and_eval(seg_path)
    prompt_frame = list(prompt.keys())[0]
    prompt_annos = prompt[prompt_frame]

    print(f"prompt: {prompt_frame}")
    print(f"eval_set: {eval_set.keys()}")

    render_path = args.model_path + f"/render{args.loaded_iter}"
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

        #prompt_frame = 26

        train_cameras = scene.getTrainCameras()
        view = [c for c in train_cameras if c.image_name == prompt_frame]
        if view == []:
            test_cameras = scene.getTestCameras()
            view = [c for c in test_cameras if c.image_name == prompt_frame]

        if view == []:
            print("no camera view for prompt frame")
            exit()

        view = view[0]


        render_pkg = render(view, gaussians, pipeline, background, args)
        render_img = render_pkg["render"].cpu()
        accum_alpha = render_pkg["accum_alpha"].cpu()


        print("background:", background)
        print(render_img.shape, render_img.max(), render_img.min())
        print(accum_alpha.shape, accum_alpha.max(), accum_alpha.min())

        # render_img shape is [3, 1080, 1440]
        # accum_alpha shape is [1080, 1440]
        # remove the background from render_img
        for i in range(3):
            render_img[i] = render_img[i] + accum_alpha * background[i].cpu()
        

        alpha_id_map = render_pkg["alpha_id_map"].cpu()
        torchvision.utils.save_image(render_img,  render_path+f"/prompt{prompt_frame}.png")

        gaussian_label = gaussians.label
        alpha_id_map = alpha_id_map.type(torch.long)
        label_map = gaussian_label[alpha_id_map]

        save_label_map = label_map.clone().type(torch.float32) / label_map.max()
        torchvision.utils.save_image(save_label_map,  render_path+f"/prompt{prompt_frame}_label_map.png")

        # save for debug
        uni_mask_map = view.mask_list[0]
        label_ids = torch.unique( uni_mask_map).type(torch.long)
        for label_id in label_ids:
            print("label_id:", label_id)
            label_map_label = label_map == label_id
            save_label_map_label = label_map_label.clone().type(torch.float32) * 255
            torchvision.utils.save_image(save_label_map_label,  render_path+f"/prompt{prompt_frame}_label_map_{label_id}.png")

            uni_label_map_label = uni_mask_map == label_id
            uni_label_map_label = uni_label_map_label.type(torch.float32) * 255
            torchvision.utils.save_image(uni_label_map_label,  render_path+f"/prompt{prompt_frame}_label_map_{label_id}_gt.png")

            if label_id == 0 or label_id == -1:
                continue
            query_id = [label_id]
            query_id = torch.tensor(query_id, dtype=torch.int32, device="cuda")
            render_pkg = render(view, gaussians, pipeline, background, args, label_id=query_id)
            torchvision.utils.save_image(render_pkg["render"],  render_path+f"/prompt{prompt_frame}_label_{label_id}.png")

        exit()


        label_map = label_map.cpu().numpy()
        label_map_cnt = np.unique(label_map, return_counts=True)
        label_map_cnt = dict(zip(label_map_cnt[0], label_map_cnt[1]))
        print("label_map_cnt:", label_map_cnt)

        # get the 3d object in gaussian model for each prompt mask
        object_to_label = {}
        for label_str, mask in prompt_annos.items():
            print("label_str:", label_str)
            mask_label_cnt = np.unique(label_map * mask, return_counts=True)
            mask_label_cnt = dict(zip(mask_label_cnt[0], mask_label_cnt[1]))
            print("mask_label_cnt:", mask_label_cnt)
            mask_label_cnt.pop(0, None)
            mask_label_cnt.pop(-1, None)
            print("mask_label_cnt:", mask_label_cnt)
            #print(mask_label_cnt)
            valid_label = [k for k, v in mask_label_cnt.items() if v > label_map_cnt[k] * 0.5]
            print("valid_label:", valid_label)
            object_to_label[label_str] = valid_label
            
            # render the 3d object 
            valid_label = torch.tensor(valid_label, dtype=torch.int32, device="cuda")
            if valid_label.shape[0] == 0:
                print("[Warning] no valid label id for label_str:", label_str)
            render_pkg = render(view, gaussians, pipeline, background, args, label_id=valid_label)
            torchvision.utils.save_image(render_pkg["render"],  render_path+f"/prompt{prompt_frame}_{label_str}.png")

            # save label_str gt iamge
            label_img = view.original_image * torch.tensor(mask).to("cuda").type(torch.bool)
            label_img = label_img + (~torch.tensor(mask).to("cuda").type(torch.bool))

            print(label_img.shape)

            torchvision.utils.save_image(label_img,  render_path+f"/prompt{prompt_frame}_{label_str}_gt.png")
            print(render_path+f"/prompt{prompt_frame}_{label_str}_gt.png")

            # save mask as image file
            mask = torch.tensor(mask, dtype=torch.float32, device="cuda")
            torchvision.utils.save_image(mask,  render_path+f"/prompt{prompt_frame}_{label_str}_mask_gt.png")
            print()
            exit()

        print("object_to_label: ", object_to_label)

            
        

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


