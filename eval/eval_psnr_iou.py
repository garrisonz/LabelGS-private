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
from utils.image_utils import psnr
import logging

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def extract_3d_object(gaussians, scene, eval_path, object_to_label, args, background):
    object_name = "Pikachu"
    test_cameras = scene.getTestCameras()
    train_cameras = scene.getTrainCameras()
    views = train_cameras + test_cameras
    # sort views by image_name
    views = sorted(views, key=lambda x: x.image_name)
    # print image_name of all views
    print("views: ", [view.image_name for view in views])
    from utils.eval_tools import interpolate_views
    views = interpolate_views(views)

    pikachu_path = eval_path + "/" + object_name
    os.makedirs(pikachu_path, exist_ok=True)
    for idx, view in tqdm(enumerate(views), desc="Rendering progress"):

        label_ids = object_to_label[object_name]
        label_ids = torch.tensor(label_ids, dtype=torch.int32, device="cuda")
        render_pkg = render(view, gaussians, pipeline, background, args, label_id=label_ids)
        render_img = render_pkg["render"]
        # idx to string with 5 digits padding with 0
        idx = '{0:05d}'.format(idx)
        torchvision.utils.save_image(render_img,  pikachu_path+f"/{idx}.png")


def eval(dataset : ModelParams, pipeline : PipelineParams, args):
    dataset.eval = True

    scene_name = args.scene_name
    print(scene_name)
    dataset_name = args.dataset_name
    print(dataset_name)

    if dataset_name == "3d_ovs":
        seg_path = f"/home/zhangyupeng/w/3drecon/LabelGS/dataset/{dataset_name}/{scene_name}/segmentations_v2"
    else:
        seg_path = f"/home/zhangyupeng/w/3drecon/LabelGS/dataset/{dataset_name}/{scene_name}/segmentations"


    from utils.eval_tools import get_prompt_and_eval
    prompt, eval_set = get_prompt_and_eval(seg_path)
    prompt_frame = list(prompt.keys())[0]
    prompt_annos = prompt[prompt_frame]

    print(f"prompt: {prompt_frame}")
    print(f"eval_set: {eval_set.keys()}")


    if dataset_name == "3d_ovs":
        eval_path = args.model_path + f"/eval{args.loaded_iter}_v2"
        eval_path = args.model_path + f"/eval{args.loaded_iter}"
    else:
        eval_path = args.model_path + f"/eval{args.loaded_iter}"
    makedirs(eval_path, exist_ok=True)

    with torch.no_grad():
        # read gaussian model
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False, mask_version=args.mask_version)
        checkpoint = os.path.join(args.model_path, f'chkpnt{args.loaded_iter}.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        train_cameras = scene.getTrainCameras()

        view = [c for c in train_cameras if c.image_name == prompt_frame][0]

        render_pkg = render(view, gaussians, pipeline, background, args)
        render_img = render_pkg["render"].cpu()
        alpha_id_map = render_pkg["alpha_id_map"].cpu()
        torchvision.utils.save_image(render_img,  eval_path+f"/prompt{prompt_frame}.png")

        gaussian_label = gaussians.label
        alpha_id_map = alpha_id_map.type(torch.long)
        label_map = gaussian_label[alpha_id_map]

        save_label_map = label_map.clone().type(torch.float32) / label_map.max()
        torchvision.utils.save_image(save_label_map,  eval_path+f"/prompt{prompt_frame}_label_map.png")


        uni_mask_map = view.mask_list[0]
        label_ids = torch.unique( uni_mask_map).type(torch.long)
        for label_id in label_ids:
            if label_id == 0:
                continue
            label_map_label = label_map == label_id
            save_label_map_label = label_map_label.clone().type(torch.float32) * 255
            torchvision.utils.save_image(save_label_map_label,  eval_path+f"/prompt{prompt_frame}_label_map_{label_id}.png")

            uni_label_map_label = uni_mask_map == label_id
            uni_label_map_label = uni_label_map_label.type(torch.float32) * 255
            torchvision.utils.save_image(uni_label_map_label,  eval_path+f"/prompt{prompt_frame}_label_map_{label_id}_gt.png")


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
            #print("mask_label_cnt:", mask_label_cnt)
            mask_label_cnt.pop(0, None)
            mask_label_cnt.pop(-1, None)
            #print("mask_label_cnt:", mask_label_cnt)
            #print(mask_label_cnt)
            valid_label = [k for k, v in mask_label_cnt.items() if v > label_map_cnt[k] * 0.5]
            #print("valid_label:", valid_label)
            object_to_label[label_str] = valid_label
            
            # render the 3d object 
            valid_label = torch.tensor(valid_label, dtype=torch.int32, device="cuda")
            if valid_label.shape[0] == 0:
                print("[Warning] no valid label id for label_str:", label_str)
            render_pkg = render(view, gaussians, pipeline, background, args, label_id=valid_label)
            torchvision.utils.save_image(render_pkg["render"],  eval_path+f"/prompt{prompt_frame}_{label_str}.png")

            # save label_str gt iamge
            label_img = view.original_image * torch.tensor(mask).to("cuda").type(torch.bool)
            #label_img = label_img + (~torch.tensor(mask).to("cuda").type(torch.bool))

            #print(label_img.shape)

            torchvision.utils.save_image(label_img,  eval_path+f"/prompt{prompt_frame}_{label_str}_gt.png")
            #print(eval_path+f"/prompt{prompt_frame}_{label_str}_gt.png")
            #exit()

            # save mask as image file
            mask = torch.tensor(mask, dtype=torch.float32, device="cuda")
            torchvision.utils.save_image(mask,  eval_path+f"/prompt{prompt_frame}_{label_str}_mask_gt.png")

            #print()

        print("object_to_label: ", object_to_label)

        #extract_3d_object(gaussians, scene, eval_path, object_to_label, args, background)
        #print()
     
        # evaluate on eval_set
        print("eval in eval_set...")
        test_cameras = scene.getTestCameras()
        test_cameras = {c.image_name: c for c in test_cameras}
        object_psnr = {}
        object_psnr_num = {}
        iou_list = []
        for frame_num, eval_anno in eval_set.items():
            print("frame_num:", frame_num)
            if frame_num not in test_cameras:
                print(f"frame_num {frame_num} not in test_cameras")
                continue
            view = test_cameras[frame_num]
            render_pkg = render(view, gaussians, pipeline, background, args)
            torchvision.utils.save_image(render_pkg["render"],  eval_path+f"/test{frame_num}.png")

            #print("eval_anno:", eval_anno.keys())
            
            for label_str, seg in eval_anno.items():
                if label_str not in object_to_label:
                    print(f"[Warning] label_str {label_str} not in object_to_label")
                    continue

                # save seg as image file
                seg = torch.tensor(seg, dtype=torch.float32, device="cuda")
                torchvision.utils.save_image(seg,  eval_path+f"/test{frame_num}_{label_str}_mask_gt.png")

                # evaluate the similarity of prompt_3d[label] and seg
                label_ids = object_to_label[label_str]
                #print(label_str, label_ids)
                label_ids = torch.tensor(label_ids, dtype=torch.int32, device="cuda")
                render_pkg = render(view, gaussians, pipeline, background, args, label_id=label_ids)
                torchvision.utils.save_image(render_pkg["render"],  eval_path+f"/test{frame_num}_{label_str}.png")

                gt = test_cameras[frame_num].original_image * seg.type(torch.bool)
                torchvision.utils.save_image(gt,  eval_path+f"/test{frame_num}_{label_str}_gt.png")

                # start. croped gt and rendering for psnr calculation
                os.makedirs(eval_path+f"/croped", exist_ok=True)
                
                from utils.eval_tools import get_bbox_from_seg2
                min_x, min_y, max_x, max_y = get_bbox_from_seg2(seg.cpu())
                min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)

                seg_croped = seg[min_x:max_x, min_y:max_y]
                torchvision.utils.save_image(seg_croped,  eval_path+f"/croped/test{frame_num}_{label_str}_mask_gt.png")

                gt_croped = gt[:, min_x:max_x, min_y:max_y]
                torchvision.utils.save_image(gt_croped,  eval_path+f"/croped/test{frame_num}_{label_str}_gt.png")
                render_img_croped = render_pkg["render"][:, min_x:max_x, min_y:max_y]
                torchvision.utils.save_image(render_img_croped,  eval_path+f"/croped/test{frame_num}_{label_str}.png")
                # end. croped gt and rendering for psnr calculation
                #psnr1 = psnr(render_img_croped, gt_croped).mean().double()

                psnr1 = psnr(render_pkg["render"], gt).mean().double()
                print(label_str, object_to_label[label_str], psnr1.item())


                # caluate iou
                mask_pred = render_pkg["render"] > 0.01
                # mask_pred is [3, 1080, 1440], convert to [1090, 1440], set each element to 1 if any of channel is 1
                mask_pred = mask_pred.any(dim=0)

                mask_pred = mask_pred.type(torch.float32)
                torchvision.utils.save_image(mask_pred,  eval_path+f"/test{frame_num}_{label_str}_mask_pred.png")

                mask_pred = np.array(mask_pred.cpu())
                mask_gt = np.array(seg.cpu())
                intersection = np.sum(np.logical_and(mask_gt, mask_pred))
                union = np.sum(np.logical_or(mask_gt, mask_pred))
                iou = (np.sum(intersection) / np.sum(union))
                print(label_str, "iou:", iou)
                iou_list.append(iou)


                if label_str not in object_psnr:
                    object_psnr[label_str] = 0
                    object_psnr_num[label_str] = 0
                object_psnr[label_str] += psnr1.item()
                object_psnr_num[label_str] += 1

            
        for label_str in object_psnr:
            object_psnr[label_str] /= object_psnr_num[label_str]
        # sort object_psnr according key
        object_psnr = dict(sorted(object_psnr.items(), key=lambda x: x[0]))

        print("object_psnr: ", object_psnr)

        # psnr_test is avg of all psnr in object_psnr
        psnr_test = sum(object_psnr.values()) / len(object_psnr)
        miou = np.array(iou_list).mean()
        summary = f"------------------ version: {args.version}, scene_name: {scene_name}, psnr_test: {psnr_test}, miou: {miou}"
        logging.info(summary)
        print(summary)


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
    parser.add_argument("--dataset_name", type=str, required=True) # example 3d_ovs
    parser.add_argument("--scene_name", type=str, required=True) # example 3d_ovs
    parser.add_argument("--mask_version", type=int, required=True)
    parser.add_argument("--version", type=int, required=True)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    log_file_name = f"{args.model_path}/eval.log"
    logging.basicConfig(filename=log_file_name, level=logging.INFO)

    eval(model.extract(args), pipeline.extract(args), args)


