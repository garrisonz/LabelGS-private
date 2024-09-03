import os
# get dataset_name from command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True) # example 3d_ovs
args = parser.parse_args()
dataset_name = args.dataset_name

dataset_path = f"/home/zhangyupeng/w/3drecon/LabelGS/dataset/{dataset_name}"
print(dataset_path)

# get folder in dataset_path dir
scene_names = os.listdir(dataset_path)
if dataset_name == "lerf_ovs":
    scene_names = [scene_name for scene_name in scene_names if "label" != scene_name]

scene_names = [scene_name for scene_name in scene_names if os.path.isdir(f"{dataset_path}/{scene_name}")]


scene_names.sort()
print("scene_names:", scene_names)

for scene_name in scene_names:

    mask_root = f"{dataset_path}/{scene_name}/mask/video_mask_auto/"

    cmd = f"python preprocess/consistent_mask/uni_mask.py --mask_root {mask_root}"

    print(cmd)
    os.system(cmd) 
    
    #exit()