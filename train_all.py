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

start_cmd = ""

iteration = 15000
version = 3
mask_version = 3

for scene_name in scene_names:

    #start_iteration = 15000
    #start_cmd = f"--start_checkpoint output/{dataset_name}/auto_{scene_name}_segEval{version}/chkpnt{start_iteration}.pth"

    cmd = (f"python train.py -s dataset/{dataset_name}/{scene_name} -m output/{dataset_name}/auto_{scene_name}_segEval{version} --eval --iteration {iteration} --label {start_cmd} --occlude_flag --mask_version {mask_version} --gpf_flag")
    print(cmd)
    os.system(cmd)
    exit()

