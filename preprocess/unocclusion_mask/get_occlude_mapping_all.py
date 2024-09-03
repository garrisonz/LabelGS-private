import os
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
scene_names.sort()



print("scene_names:", scene_names)

version = 1
mask_version = 3

for scene_name in scene_names:

    if scene_name in ["bicycle", "bonsai"]:
        continue

    cmd = (f"python preprocess/unocclusion_mask/get_occlude_mapping.py --scene_path dataset/{dataset_name}/{scene_name} --out_dir output/{dataset_name}/auto_{scene_name}_segEval{version} --mask_version {mask_version}")
    print(cmd)
    os.system(cmd)

    #exit()

