# LabelGS: Label-Aware 3D Gaussian Splatting for 3D Scene Segmentation

Yupeng Zhang, et al. 2024

\| Webpage(TBA) \| Full Paper(TBA) \|

----

## 1. Dataset

- We provide `segmentations` folder for nerf_llff_data dataset and `segmentations_v2` for 3d_ovs dataset, to evaluate the performance of 3D Object segmentation by extracting 3D representation primitive

## 2. Data Preprocess

### 2.1 Densify Views

- native 3DGS reconstruction
  
  ```jsx
  python preprocess/densify_views/train_3dgs_all.py --dataset_name 3d_ovs
  ```

- render dense views as video frames from 3DGS model
  
  ```jsx
  python preprocess/densify_views/render_video_all.py --dataset_name 3d_ovs
  ```

### 2.2 Cross-View Consistent Mask

- Build a python environment according [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA)

- generate cross-view consistent masks from DEVA in DEVA env
  
  ```jsx
  (deva) python preprocess/consistent_mask/demo_automatic_all.py --dataset_name 3d_ovs
  ```

- Mapping RGB mask to gray mask
  
  ```jsx
  python preprocess/consistent_mask/uni_mask_all.py --dataset_name 3d_ovs 
  ```

### 2.3 Unocclusion Mask

- Build a python env for [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2)

- Generate depth map in DepthAnythingV2 env
  
  ```jsx
  (depth_anything_v2) python preprocess/unocclusion_mask/run_all.py --dataset_name 3d_ovs
  ```

- Obtain occlusion relationship
  
  ```jsx
  python preprocess/unocclusion_mask/get_occlude_mapping_all.py --dataset_name 3d_ovs
  ```

## 3. Training

- Label-Aware 3D Gaussian Splatting
  
  ```jsx
  Python train_all.py  --dataset_name 3d_ovs
  ```

## 4. GUI
  ```
  python labelgs_gui.py -m {model_output_path} -s {scene_path}
  ```
  For example, `python labelgs_gui.py -m output/3d_ovs/auto_bed_segEval3 -s dataset/3d_ovs/bed`


Segmenting one object.

https://github.com/user-attachments/assets/ebd8f2d8-0f27-49ee-a67c-c11bfb454479

Segmentint multiple objects simultaneously.

https://github.com/user-attachments/assets/f557a4cf-7b4f-4254-9f21-ec0119161a7b

Segmenting the occluded object and showing the occluded region.

https://github.com/user-attachments/assets/b5ff5558-f3a1-4ae2-b8d7-4c611372f3a9


## 5. Evaluation

- Evaluation for PSNR and mIoU
  
  ```jsx
  python eval/eval_all.py --dataset_name 3d_ovs
  ```

---
