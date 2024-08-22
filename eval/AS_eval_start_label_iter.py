import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True) # example 3d_ovs
parser.add_argument("--white_background", action="store_true", default=False) # example 3d_ovs
args = parser.parse_args()
dataset_name = args.dataset_name
white_background = args.white_background

dataset_path = f"/home/zhangyupeng/w/3drecon/LabelGS/dataset/{dataset_name}"
print(dataset_path)

loaded_iter = 15000
mask_version = 3

if white_background:
    white_background = "--white_background"
else:
    white_background = ""

output_dataset_name = dataset_name


scene_name = "bed"
start_version = 43

output_list = []

for v in range(1):
    version = start_version + v

    output = f"output/{output_dataset_name}/auto_{scene_name}_segEval{version}"
    output_list.append(output)

    cmd = (f"python eval_psnr_iou.py -m {output} --loaded_iter {loaded_iter} {white_background} --dataset_name {dataset_name} --scene_name {scene_name} --mask_version {mask_version} --version {version}")
    print(cmd)
    os.system(cmd)
    print("")
    #break

print(output_list)

for output in output_list:
    log_file_path = f"{output}/eval.log"
    print(log_file_path)
    if os.path.exists(log_file_path):
        # print the last line of the log file
        with open(log_file_path, "r") as f:
            lines = f.readlines()
            print(lines[0])



