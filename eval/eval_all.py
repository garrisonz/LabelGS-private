import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True) # example 3d_ovs
parser.add_argument("--white_background", action="store_true", default=False) # example 3d_ovs
parser.add_argument("--nas_output", action="store_true", default=False)
args = parser.parse_args()
dataset_name = args.dataset_name
white_background = args.white_background
nas_output = args.nas_output

dataset_path = f"/home/zhangyupeng/w/3drecon/LabelGS/dataset/{dataset_name}"
print(dataset_path)

# get folder in dataset_path dir
scene_names = os.listdir(dataset_path)
if dataset_name == "lerf_ovs":
    scene_names = [scene_name for scene_name in scene_names if "label" != scene_name]
scene_names.sort()
print("scene_names:", scene_names)

loaded_iter = 15000
version = 3
mask_version = 3

if white_background:
    white_background = "--white_background"
else:
    white_background = ""

output_dataset_name = dataset_name
if nas_output:
    output_dataset_name = "nas_" + dataset_name

#scene_names = ["sofa", "table"]

for scene_name in scene_names:
#    scene_name = "blue_sofa"

    cmd = (f"python -m eval.eval_psnr_iou -m output/{output_dataset_name}/auto_{scene_name}_segEval{version} --loaded_iter {loaded_iter} {white_background} --dataset_name {dataset_name} --scene_name {scene_name} --mask_version {mask_version} --version {version}")
    print(cmd)
    os.system(cmd)
    print("")
    exit()

