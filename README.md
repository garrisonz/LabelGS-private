# LabelGS: Label-Aware 3D Gaussian Splatting for 3D Scene Segmentation

Yupeng Zhang, et al. 2024

\| Webpage(TBA) \| Full Paper(TBA) \| Video(TBA) \|

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

- Build a python environment according DEVA

- generate cross-view consistent masks from DEVA
  
  ```jsx
  python preprocess/consistent_mask/demo_automatic_all.py --dataset_name 3d_ovs
  ```

- Mapping RGB mask to gray mask
  
  ```jsx
  python preprocess/consistent_mask/uni_mask_all.py --dataset_name 3d_ovs 
  ```

### 2.3 Unocclusion Mask

- Build a python env for DepthAnythingV2

- Generate depth map
  
  ```jsx
  python preprocess/unocclusion_mask/run_all.py --dataset_name 3d_ovs
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

  

https://github.com/user-attachments/assets/ebd8f2d8-0f27-49ee-a67c-c11bfb454479




## 5. Evaluation

- Evaluation for PSNR and mIoU
  
  ```jsx
  python eval/eval_all.py --dataset_name 3d_ovs
  ```

---
