#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import utils.lift_label as lift_label
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
TENSORBOARD_FOUND = False
    

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, occlude_flag, mask_version, gpf_flag, start_label_iter, args):
    print("occlude_flag: ", occlude_flag)

    output_dir = dataset.model_path
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, mask_version=mask_version)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f"restored gaussian: {gaussians.get_xyz.shape}, {gaussians.label.shape}")
        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    print(f"start train from {first_iter}")
    gui_label_idx = -1
    pre_do_training = False
    label_id = -1
    last_reset_opacity_iter = 0
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    # unique value of gaussian.label as mask_id list
                    mask_ids = torch.unique(gaussians.label)
                    # remove -1 element from mask_ids
                    mask_ids = mask_ids[mask_ids != -1]
                    # sort mask_ids 
                    mask_ids = torch.sort(mask_ids)[0]
                    # mask_ids is a list of mask_id
                    if pre_do_training != do_training:
                        pre_do_training = do_training
                        gui_label_idx = (gui_label_idx + 1) % (len(mask_ids)+1)
                        if gui_label_idx < len(mask_ids):
                            label_id = mask_ids[gui_label_idx].item()
                        else:
                            label_id = -1
    
                        print("render label_id:", label_id)

                    label_id = torch.tensor([-1], dtype=torch.int32, device="cuda")
                    net_image = render(custom_cam, gaussians, pipe, background, opt, scaling_modifer, label_id=label_id)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        bg = torch.rand((3), device="cuda") if opt.random_background else background 

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, opt)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # Loss
        start_label_iter = start_label_iter
        if iteration == start_label_iter:
            print(f"[{iteration}] Start label loss!!!")
        label_loss = 0
        if iteration >= start_label_iter and args.label:
            alpha_max_map = render_pkg["alpha_map"]
            alpha_id_map = render_pkg["alpha_id_map"]
            proj_means2D = render_pkg["proj_means2D"]

            proj_means2D = proj_means2D.long()

            uni_mask_map = viewpoint_cam.mask_list[0]
            mask_ids = torch.unique(uni_mask_map)

            train_path = output_dir + "/train"
            os.makedirs(train_path, exist_ok=True)
            image_name = viewpoint_cam.image_name

            # #start save
            save_debug_interval = 1000
            # if iteration % save_debug_interval == 0:
            #     save_image = image.clone().type(torch.float32)
            #     torchvision.utils.save_image(save_image,  train_path+f"/{iteration}_{image_name}_image.png")
            
            #     label_map = gaussians.label[alpha_id_map.type(torch.long)]
            #     save_label_map = label_map.clone().type(torch.float32) / label_map.max()
            #     torchvision.utils.save_image(save_label_map,  train_path+f"/{iteration}_{image_name}_label_map_before.png")

            #     save_uni_mask_map = uni_mask_map.clone().type(torch.float32) / uni_mask_map.max()
            #     torchvision.utils.save_image(save_uni_mask_map,  train_path+f"/{iteration}_{image_name}_uni_mask_map.png")
            # # end save

            sample_mask_ids = mask_ids[torch.randperm(mask_ids.shape[0])][:args.mask_sample_number]

            # sort for debug
            # sample_mask_ids = torch.sort(sample_mask_ids)[0]

            for i, mask_id in enumerate(sample_mask_ids):
                if mask_id == 0:
                    continue
                label_id = mask_id
                mask_map = uni_mask_map == mask_id

                alpha_max_threshold = args.alpha_max_threshold

                lift_label.lift_label(gaussians.label, alpha_max_map, alpha_id_map, mask_map, proj_means2D, label_id, opt, iteration, output_dir, alpha_max_threshold=alpha_max_threshold, gpf_flag=gpf_flag)

                if (gaussians.label==label_id).shape[0] == 0:
                    print(f"[Warning]no valid label_id: {label_id} in gaussians ")
                    continue

                label_gt = mask_map * viewpoint_cam.original_image.cuda()

                tensor_label_id = torch.tensor([label_id], dtype=torch.int32, device="cuda")
                label_render_pkg = render(viewpoint_cam, gaussians, pipe, bg, opt, label_id=tensor_label_id)
                label_image = label_render_pkg["render"]

                if occlude_flag:
                    # start get unoccluded region
                    occlude_map = torch.zeros_like(mask_map)

                    occlude_labels = viewpoint_cam.occlude_mapping[label_id.item()]
                    for occlude_label in occlude_labels:
                        occlude_map += uni_mask_map == occlude_label
                    unocclude_map = occlude_map == 0
                    # end get unoccluded region

                    # save skip_map as image file
                    if iteration % save_debug_interval == 0:
                        save_filter_map = unocclude_map.clone().type(torch.float32) / unocclude_map.max()
                        torchvision.utils.save_image(save_filter_map,  train_path+f"/{iteration}_{image_name}_label_{label_id.item()}_unocclude.png")
                else:
                    unocclude_map = torch.ones_like(mask_map)

                # # save label_image as image file
                # if iteration % save_debug_interval == 0:
                #     save_label_image = label_image.clone().type(torch.float32)
                #     torchvision.utils.save_image(save_label_image,  train_path+f"/{iteration}_{image_name}_label_{label_id.item()}_render.png")
                #     save_label_gt = label_gt.clone().type(torch.float32)
                #     torchvision.utils.save_image(save_label_gt,  train_path+f"/{iteration}_{image_name}_label_{label_id.item()}_gt.png")

                #label_gt_unocc = label_gt * unocclude_map
                #torchvision.utils.save_image(label_gt_unocc,  train_path+f"/{iteration}_{image_name}_label_{label_id.item()}_gt_unocc.png")

                #label_image_unocc = label_image * unocclude_map
                #torchvision.utils.save_image(label_image_unocc,  train_path+f"/{iteration}_{image_name}_label_{label_id.item()}_image_unocc.png")

                #torchvision.utils.save_image(label_image,  train_path+f"/{iteration}_{image_name}_label_{label_id.item()}_image.png")

                label_loss += l1_loss(label_image * unocclude_map, label_gt * unocclude_map)

            # # start save
            # if iteration % save_debug_interval == 0:
            #     label_map = gaussians.label[alpha_id_map.type(torch.long)]
            #     save_label_map = label_map.clone().type(torch.float32) / label_map.max()
            #     torchvision.utils.save_image(save_label_map,  train_path+f"/{iteration}_{image_name}_label_map_after.png")

            # end save

            # if iteration % 10 == 0:
            #     exit()

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + label_loss

        loss.backward()
        iter_end.record()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, opt))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, scene, pipe, bg)
                    #print(f"\n[Warning][ITER {iteration}] densify_and_prune")
                
                if (iteration % opt.opacity_reset_interval == 0 and (opt.iterations - iteration > opt.opacity_reset_interval * 0.5)) or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    print(f"\n[Warning][ITER {iteration}] reset_opacity")

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def create_test_iterations():
    iters = []
    i = 1000
    while i <= args.iterations:
        iters.append(i)
        i += 500

    return iters

def create_save_iterations():
    iters = [7000, 10000, 15000]
    i = 20000
    while i <= args.iterations:
        iters.append(i)
        i += 10000

    return iters

def create_checkpoint_iterations():
    iters = []
    i = 1000
    while i <= args.iterations:
        iters.append(i)
        i += 1000

    return iters

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=55555)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    #parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10000, 15000, 15010, 20000, 30_000, 40000, 50000, 60000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=None)
    parser.add_argument("--save_image_iterations", nargs="+", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7000, 10000, 15000, 20000, 30000, 40000, 50000, 60000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--label", action="store_true", default=False)
    parser.add_argument("--occlude_flag", action="store_true", default=False)
    parser.add_argument("--gpf_flag", action="store_true", default=False)
    parser.add_argument("--mask_version", type=int, default=3)
    parser.add_argument('--start_label_iter', type=int, default=1000)
    parser.add_argument('--alpha_max_threshold', type=float, default=0.6)
    parser.add_argument('--mask_sample_number', type=int, default=10)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if args.test_iterations is None:
        args.test_iterations = create_test_iterations()

    if args.save_image_iterations is None:
        args.save_image_iterations = create_test_iterations()

    if args.save_iterations is None:
        args.save_iterations = create_test_iterations()

#    args.checkpoint_iterations = create_checkpoint_iterations()


    if args.gpf_flag == False:
        print("[Warning]not use GPF")

    print("alpha_max_threshold:", args.alpha_max_threshold)
    print("mask_sample_number:", args.mask_sample_number)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.occlude_flag, args.mask_version, args.gpf_flag, args.start_label_iter, args)

    # All done
    print("\nTraining complete.")
