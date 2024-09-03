import os
import cv2
import numpy as np

# get scene_path from command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--max_hight', type=int, default=540)
parser.add_argument('--image_dir', type=str, required=True) # "dataset/3d_ovs/sofa"
parser.add_argument('--output_dir', type=str, required=True) # "dataset/3d_ovs/sofa"

args = parser.parse_args()
max_hight = args.max_hight
image_dir = args.image_dir
output_dir = args.output_dir

def downsample_image(source_folder, output_folder, is_mask=False):
    img_dir = source_folder
    output_folder = output_folder
    os.makedirs(output_folder, exist_ok=True)

    #remove img_dir_8
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print(img_dir)
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        if not img_name.endswith(".jpg") and not img_name.endswith(".png") and not img_name.endswith(".JPG"):
            continue
        img = cv2.imread(img_path)

        image = img
        orig_w, orig_h = image.shape[1], image.shape[0]
        if orig_h > max_hight:
            global_down = orig_h / max_hight
        else:
            global_down = 1
        scale = float(global_down)
        resolution = (int( orig_w  / scale), int(orig_h / scale))
        image = cv2.resize(image, resolution)
        img = image

        if is_mask:
            img = img > 128
            img = img.astype(np.uint8) * 255

        cv2.imwrite(os.path.join(output_folder, img_name), img)
        print(f"downsampled {img_name} to {output_folder} {img.shape}")


downsample_image(image_dir, output_dir)
