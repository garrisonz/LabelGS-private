import os
# get dataset_name from command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True) # example 3d_ovs
parser.add_argument("--eval", action="store_true")
args = parser.parse_args()
dataset_name = args.dataset_name
eval = args.eval

dataset_path = f"/home/zhangyupeng/w/3drecon/LabelGS/dataset/{dataset_name}"
print(dataset_path)

# get folder in dataset_path dir
scene_names = os.listdir(dataset_path)
if dataset_name == "lerf_ovs":
    scene_names = [scene_name for scene_name in scene_names if "label" != scene_name]
scene_names.sort()

# skip if scene_name is not a folder
scene_names = [scene_name for scene_name in scene_names if os.path.isdir(f"{dataset_path}/{scene_name}")]
print("scene_names:", scene_names)

start_cmd = ""

eval_cmd = ""
if eval:
    eval_cmd = "--eval"

iteration = 30000

for scene_name in scene_names:

    #scene_name = "garden"


    #start_iteration = 15000
    #start_cmd = f"--start_checkpoint output/{dataset_name}/{scene_name}/chkpnt{start_iteration}.pth"

    cmd = (f"python train.py -s dataset/{dataset_name}/{scene_name} -m output/{dataset_name}/{scene_name} {eval_cmd} --iteration {iteration} {start_cmd}")
    print(cmd)
    os.system(cmd)
#    exit()
