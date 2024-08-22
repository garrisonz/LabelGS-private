import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True) # example 3d_ovs
args = parser.parse_args()
dataset_name = args.dataset_name

dataset_path = f"/home/zhangyupeng/w/3drecon/LangSplat/dataset/{dataset_name}"
print(dataset_path)

# get folder in dataset_path dir
scene_names = os.listdir(dataset_path)
if dataset_name == "lerf_ovs":
    scene_names = [scene_name for scene_name in scene_names if "label" != scene_name]
scene_names.sort()
print("scene_names:", scene_names)

for scene_name in scene_names:

    cmd = (f"python run.py --encoder vitl --img-path ~/w/3drecon/data/{dataset_name}/{scene_name}/images/ --outdir ~/w/3drecon/data/{dataset_name}/{scene_name}/depth/ --pred-only --grayscale")
    print(cmd)
    os.system(cmd)
    #exit()

