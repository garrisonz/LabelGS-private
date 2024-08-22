import os
import cv2
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True) # example 3d_ovs
args = parser.parse_args()
dataset_name = args.dataset_name

dataset_path = f"/home/zhangyupeng/w/3drecon/LabelGS/dataset/{dataset_name}"
print(dataset_path)

# get folder in dataset_path dir
scene_names = os.listdir(dataset_path)
scene_names.sort()
print("scene_names:", scene_names)

start_cmd = ""

iteration = 15000
version = 6
mask_version = 3


def crop_save_image(pred_path, min_x, min_y, max_x, max_y, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not os.path.exists(pred_path):
        print(f"pred_path does not exist: {pred_path}")
        return

    pred_image = cv2.imread(pred_path)
    pred_image_croped = pred_image[min_x:max_x, min_y:max_y]
    cv2.imwrite(save_path, pred_image_croped)
    print(save_path)


for scene_name in scene_names:

    seg_path = f"/home/zhangyupeng/w/3drecon/LabelGS/dataset/{dataset_name}/{scene_name}/segmentations_v2"

    eval_path = f"output/{dataset_name}/auto_{scene_name}_segEval{version}/eval{iteration}"


    from utils.eval_tools import get_prompt_and_eval
    _, eval_set = get_prompt_and_eval(seg_path)
    for frame_num, eval_anno in eval_set.items():
        for label_str, seg in eval_anno.items():

            from utils.eval_tools import get_bbox_from_seg2
            min_x, min_y, max_x, max_y = get_bbox_from_seg2(seg, 0.15)
            min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)


            pred_path = f"{eval_path}/test{frame_num}_{label_str}.png"
            save_path = f"{eval_path}/croped/test{frame_num}_{label_str}.png"
            crop_save_image(pred_path, min_x, min_y, max_x, max_y, save_path)

            gt_path = f"{eval_path}/test{frame_num}_{label_str}_gt.png"
            save_path = f"{eval_path}/croped/test{frame_num}_{label_str}_gt.png"
            crop_save_image(gt_path, min_x, min_y, max_x, max_y, save_path)
            #exit()


    #exit()