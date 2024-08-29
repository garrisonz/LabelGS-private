import os

# get dataset_name from command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True) # example 3d_ovs
args = parser.parse_args()
dataset_name = args.dataset_name

project_path = "/home/zhangyupeng/w/3drecon/LabelGS"
dataset_path = f"{project_path}/dataset/{dataset_name}"
print(dataset_path)

# get folder in dataset_path dir
scene_names = os.listdir(dataset_path)

if dataset_name == "lerf_ovs":
    scene_names = [scene_name for scene_name in scene_names if "label" != scene_name]

scene_names.sort()
print("scene_names:", scene_names)

for scene_name in scene_names:
    scene_name = "garden"

    img_path = f"{dataset_path}/{scene_name}/mask/video_mask_auto/Annotations"
    mask_root = f"{dataset_path}/{scene_name}/mask/video_mask_auto2"

    cmd = f"python preprocess/consistent_mask/demo_automatic.py --chunk_size 4 --img_path {img_path} --amp --temporal_setting semionline --size 480 --output {mask_root}"

    print(cmd)
    os.system(cmd) 
    exit()