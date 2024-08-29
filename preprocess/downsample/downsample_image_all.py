import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, required=True) # "3d_ovs"
args = parser.parse_args()
dataset_name = args.dataset_name

dataset_path = f"/home/zhangyupeng/w/3drecon/LabelGS/dataset/{dataset_name}"

# get folder in dataset_path dir
scene_names = os.listdir(dataset_path)

# remove file in scnen_names
scene_names = [scene_name for scene_name in scene_names if os.path.isdir(f"{dataset_path}/{scene_name}")]
print("scene_names:", scene_names)

#scene_names = ['bench', 'bed', 'snacks', 'table', 'lawn', 'blue_sofa', 'covered_desk', 'office_desk']

scene_names.sort()

for scene_name in scene_names:
    if scene_name == "bicycle":
        continue

    scene_path = f"{dataset_path}/{scene_name}"
    print("downscale: ", scene_path)

    os.system(f"python preprocess/downsample/downsample_image2.py --scene_path {scene_path}") 
    #exit()