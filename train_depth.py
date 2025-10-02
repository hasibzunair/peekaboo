# Code for Peekaboo
# Author: Hasib Zunair
# Modified from https://github.com/valeoai/FOUND

"""Training code for Peekaboo with Depth Prediction"""

import os
import sys
import json
import argparse

import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from model import PeekabooModel
from evaluation.saliency import evaluate_saliency
from misc import batch_apply_bilateral_solver, set_seed, load_config, Logger

from datasets.datasets import build_dataset


def get_argparser():
    parser = argparse.ArgumentParser(
        description="Training of Peekaboo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--exp-name", type=str, default=None, help="Exp name.")
    parser.add_argument(
        "--log-dir", type=str, default="outputs", help="Logging and output directory."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Root directories of training and evaluation datasets.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/peekaboo_DUTS-TR.yaml",
        help="Path of config file.",
    )
    parser.add_argument(
        "--save-model-freq", type=int, default=250, help="Frequency of model saving."
    )
    parser.add_argument(
        "--visualization-freq",
        type=int,
        default=10,
        help="Frequency of prediction visualization in tensorboard.",
    )
    # NEW: Depth training arguments
    parser.add_argument(
        "--enable-depth",
        action="store_true",
        help="Enable depth prediction task head.",
    )
    parser.add_argument(
        "--depth-loss-weight",
        type=float,
        default=1.0,
        help="Weight for depth prediction loss.",
    )

    args = parser.parse_args()
    return args


def train_model(
    model,
    config,
    dataset,
    dataset_dir,
    visualize_freq=10,
    save_model_freq=500,
    tensorboard_log_dir=None,
    enable_depth=False,  # NEW
    depth_loss_weight=1.0,  # NEW
):

    # Diverse
    print(f"Data will be saved in {tensorboard_log_dir}")
    save_dir = tensorboard_log_dir
    if tensorboard_log_dir is not None:
        # Logging
        if not os.path.exists(tensorboard_log_dir):
            os.makedirs(tensorboard_log_dir)
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(tensorboard_log_dir)

    # Deconvolution, train only the decoder(s)
    sigmoid = nn.Sigmoid()
    model.decoder.train()
    model.decoder.to("cuda")

    # NEW: Set depth decoder to train mode if enabled
    if enable_depth:
        model.depth_decoder.train()
        model.depth_decoder.to("cuda")
        print(f"Depth prediction enabled with loss weight: {depth_loss_weight}")

    ################################################################################
    #                                                                              #
    #                      Setup loss, optimizer and scheduler                     #
    #                                                                              #
    ################################################################################

    criterion = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()
    
    # NEW: Depth loss - using L1 loss for depth prediction (common choice)
    criterion_depth_l1 = nn.L1Loss()
    criterion_depth_mse = nn.MSELoss()
    
    # NEW: Collect parameters from both decoders
    params_to_optimize = list(model.decoder.parameters())
    if enable_depth:
        params_to_optimize += list(model.depth_decoder.parameters())
    
    optimizer = torch.optim.AdamW(params_to_optimize, lr=config.training["lr0"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.training["step_lr_size"],
        gamma=config.training["step_lr_gamma"],
    )

    ################################################################################
    #                                                                              #
    #                                Dataset                                       #
    #                                                                              #
    ################################################################################

    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.training["batch_size"], shuffle=True, num_workers=2
    )

    ################################################################################
    #                                                                              #
    #                                Training loop                                 #
    #                                                                              #
    ################################################################################

    n_iter = 0
    for epoch in range(config.training["nb_epochs"]):
        running_loss = 0.0
        running_seg_loss = 0.0  # NEW: Track segmentation loss separately
        running_depth_loss = 0.0  # NEW: Track depth loss separately
        tbar = tqdm(enumerate(trainloader, 0), leave=None)
        for i, data in tbar:

            # Get the inputs
            # NEW: Unpack depth_gt if available (assuming dataset returns it)
            if enable_depth:
                inputs, masked_inputs, _, input_nonorm, masked_input_nonorm, _, _, depth_gt = data
                depth_gt = depth_gt.to("cuda")
            else:
                inputs, masked_inputs, _, input_nonorm, masked_input_nonorm, _, _ = data

            ######## For debug #######
            # def to_img(ten):
            #     #ten =(input_nonorm[0].permute(1,2,0).detach().cpu().numpy()+1)/2
            #     ten =(ten.permute(1,2,0).detach().cpu().numpy())
            #     ten=(ten*255).astype(np.uint8)
            #     #ten=cv2.cvtColor(ten,cv2.COLOR_RGB2BGR)
            #     return ten
            # import pdb; pdb.set_trace()
            # im = to_img(input_nonorm[0])
            # plt.imshow(im); plt.show()

            # Inputs and masked inputs
            inputs = inputs.to("cuda")
            masked_inputs = masked_inputs.to("cuda")

            # zero the parameter gradients
            optimizer.zero_grad()

            ################################################################################
            #                                                                              #
            #                                Unsupervised Segmenter                        #
            #                                                                              #
            ################################################################################

            # Get predictions (now returns depth as well if enabled)
            if enable_depth:
                preds, depth_pred = model(inputs, return_depth=True)
            else:
                preds = model(inputs)
                
            # Binarization
            preds_mask = (sigmoid(preds.detach()) > 0.5).float()
            # Apply bilateral solver
            preds_mask_bs, _ = batch_apply_bilateral_solver(data, preds_mask.detach())
            # Flatten
            flat_preds = preds.permute(0, 2, 3, 1).reshape(-1, 1)

            #### Compute unsupervised segmenter loss ####
            alpha = 1.5
            preds_bs_loss = alpha * criterion(
                flat_preds, preds_mask_bs.reshape(-1).float()[:, None]
            )
            print(preds_bs_loss)
            writer.add_scalar("Loss/L_seg", preds_bs_loss, n_iter)
            loss = preds_bs_loss
            seg_loss = preds_bs_loss.item()  # NEW: Track segmentation loss

            ################################################################################
            #                                                                              #
            #                            Masked Feature Predictor (MFP)                    #
            #                                                                              #
            ################################################################################

            # Get predictions
            if enable_depth:
                preds_mfp, depth_pred_mfp = model(masked_inputs, return_depth=True)
            else:
                preds_mfp = model(masked_inputs)
                
            # Binarization
            preds_mask_mfp = (sigmoid(preds_mfp.detach()) > 0.5).float()
            # Apply bilateral solver
            preds_mask_mfp_bs, _ = batch_apply_bilateral_solver(
                data, preds_mask_mfp.detach()
            )
            # Flatten
            flat_preds_mfp = preds_mfp.permute(0, 2, 3, 1).reshape(-1, 1)

            #### Compute masked feature predictor loss ####
            beta = 1.0
            preds_bs_cb_loss = beta * criterion(
                flat_preds_mfp, preds_mask_mfp_bs.reshape(-1).float()[:, None]
            )
            writer.add_scalar("Loss/L_mfp", preds_bs_cb_loss, n_iter)
            loss += preds_bs_cb_loss
            seg_loss += preds_bs_cb_loss.item()  # NEW: Track segmentation loss

            ################################################################################
            #                                                                              #
            #                       Predictor Consistency Loss (PCL)                       #
            #                                                                              #
            ################################################################################

            gamma = 1.0
            task_sim_loss = gamma * criterion_mse(
                preds_mask_bs.reshape(-1).float()[:, None],
                preds_mask_mfp_bs.reshape(-1).float()[:, None],
            )
            writer.add_scalar("Loss/L_pcl", task_sim_loss, n_iter)
            loss += task_sim_loss
            seg_loss += task_sim_loss.item()  # NEW: Track segmentation loss

            ### Compute loss between soft masks and their binarized versions ####
            self_loss = criterion(flat_preds, preds_mask.reshape(-1).float()[:, None])

            self_loss = self_loss * 4.0
            loss += self_loss
            writer.add_scalar("Loss/L_regularization", self_loss, n_iter)
            seg_loss += self_loss.item()  # NEW: Track segmentation loss

            ################################################################################
            #                                                                              #
            #                       NEW: Depth Prediction Loss                             #
            #                                                                              #
            ################################################################################

            if enable_depth:
                # Resize depth predictions to match ground truth size if needed
                if depth_pred.shape != depth_gt.shape:
                    depth_pred_resized = nn.functional.interpolate(
                        depth_pred, size=depth_gt.shape[-2:], 
                        mode='bilinear', align_corners=False
                    )
                    if depth_pred_mfp.shape != depth_gt.shape:
                        depth_pred_mfp_resized = nn.functional.interpolate(
                            depth_pred_mfp, size=depth_gt.shape[-2:],
                            mode='bilinear', align_corners=False
                        )
                    else:
                        depth_pred_mfp_resized = depth_pred_mfp
                else:
                    depth_pred_resized = depth_pred
                    depth_pred_mfp_resized = depth_pred_mfp

                # Main depth loss (L1 + MSE combination for robust training)
                depth_loss_l1 = criterion_depth_l1(depth_pred_resized, depth_gt)
                depth_loss_mse = criterion_depth_mse(depth_pred_resized, depth_gt)
                depth_loss_main = 0.7 * depth_loss_l1 + 0.3 * depth_loss_mse
                
                # Depth loss for masked input (helps regularization)
                depth_loss_mfp_l1 = criterion_depth_l1(depth_pred_mfp_resized, depth_gt)
                depth_loss_mfp_mse = criterion_depth_mse(depth_pred_mfp_resized, depth_gt)
                depth_loss_mfp = 0.7 * depth_loss_mfp_l1 + 0.3 * depth_loss_mfp_mse
                
                # Consistency loss between depth predictions
                depth_consistency_loss = criterion_depth_mse(
                    depth_pred_resized, depth_pred_mfp_resized.detach()
                )
                
                # Combined depth loss
                total_depth_loss = (
                    depth_loss_main + 
                    0.5 * depth_loss_mfp + 
                    0.3 * depth_consistency_loss
                )
                
                # Weight and add to total loss
                weighted_depth_loss = depth_loss_weight * total_depth_loss
                loss += weighted_depth_loss
                
                # Log depth losses
                writer.add_scalar("Loss/depth_main", depth_loss_main, n_iter)
                writer.add_scalar("Loss/depth_mfp", depth_loss_mfp, n_iter)
                writer.add_scalar("Loss/depth_consistency", depth_consistency_loss, n_iter)
                writer.add_scalar("Loss/depth_total_weighted", weighted_depth_loss, n_iter)
                
                running_depth_loss += weighted_depth_loss.item()

            ################################################################################
            #                                                                              #
            #                       Update weights and scheduler step                      #
            #                                                                              #
            ################################################################################

            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/total_loss", loss, n_iter)
            writer.add_scalar("params/lr", optimizer.param_groups[0]["lr"], n_iter)
            scheduler.step()

            ################################################################################
            #                                                                              #
            #                       Visualize predictions and show stats                   #
            #                                                                              #
            ################################################################################

            # Visualize predictions in tensorboard
            if n_iter % visualize_freq == 0:
                # images and predictions
                grid = torchvision.utils.make_grid(input_nonorm[:5])
                writer.add_image("training/images", grid, n_iter)
                p_grid = torchvision.utils.make_grid(preds_mask[:5])
                writer.add_image("training/preds", p_grid, n_iter)

                # masked images and predictions
                m_grid = torchvision.utils.make_grid(masked_input_nonorm[:5])
                writer.add_image("training/masked_images", m_grid, n_iter)
                mp_grid = torchvision.utils.make_grid(preds_mask_mfp[:5])
                writer.add_image("training/masked_preds", mp_grid, n_iter)
                
                # NEW: Visualize depth predictions if enabled
                if enable_depth:
                    # Normalize depth for visualization
                    depth_vis = (depth_pred_resized[:5] - depth_pred_resized[:5].min()) / (
                        depth_pred_resized[:5].max() - depth_pred_resized[:5].min() + 1e-8
                    )
                    d_grid = torchvision.utils.make_grid(depth_vis)
                    writer.add_image("training/depth_pred", d_grid, n_iter)
                    
                    # Visualize ground truth depth
                    depth_gt_vis = (depth_gt[:5] - depth_gt[:5].min()) / (
                        depth_gt[:5].max() - depth_gt[:5].min() + 1e-8
                    )
                    dgt_grid = torchvision.utils.make_grid(depth_gt_vis)
                    writer.add_image("training/depth_gt", dgt_grid, n_iter)
                    
            # Statistics
            running_loss += loss.item()
            running_seg_loss += seg_loss
            
            # NEW: Enhanced statistics with depth
            if enable_depth:
                tbar.set_description(
                    f"{dataset.name}| train | iter {n_iter} | "
                    f"loss: {running_loss / (i + 1):.3f} | "
                    f"seg: {running_seg_loss / (i + 1):.3f} | "
                    f"depth: {running_depth_loss / (i + 1):.3f}"
                )
            else:
                tbar.set_description(
                    f"{dataset.name}| train | iter {n_iter} | loss: ({running_loss / (i + 1):.3f}) "
                )

            ################################################################################
            #                                                                              #
            #                           Save model and evaluate                            #
            #                                                                              #
            ################################################################################

            # Save model
            if n_iter % save_model_freq == 0 and n_iter > 0:
                model.decoder_save_weights(save_dir, n_iter)

            # Evaluation
            if n_iter % config.evaluation["freq"] == 0 and n_iter > 0:
                for dataset_eval_name in config.evaluation["datasets"]:
                    val_dataset = build_dataset(
                        root_dir=dataset_dir,
                        dataset_name=dataset_eval_name,
                        for_eval=True,
                        dataset_set=None,
                    )
                    evaluate_saliency(
                        val_dataset, model=model, n_iter=n_iter, writer=writer
                    )

            if n_iter == config.training["max_iter"]:
                model.decoder_save_weights(save_dir, n_iter)
                print("\n----" "\nTraining done.")
                writer.close()
                return model

            n_iter += 1

        print(f"##### Number of epoch is {epoch} and n_iter is {n_iter} #####")

    # Save model
    model.decoder_save_weights(save_dir, n_iter)
    print("\n----" "\nTraining done.")
    writer.close()
    return model


def main():

    ########## Get arguments ##########

    args = get_argparser()

    ########## Setup ##########

    # Load config yaml file
    config, config_ = load_config(args.config)

    # Experiment name
    exp_name = "{}-{}{}".format(
        config.training["dataset"], config.model["arch"], config.model["patch_size"]
    )

    if args.exp_name is not None:
        exp_name = f"{args.exp_name}-{exp_name}"
    
    # NEW: Add depth suffix to experiment name if enabled
    if args.enable_depth:
        exp_name = f"{exp_name}-depth"

    # Log dir
    output_dir = os.path.join(args.log_dir, exp_name)

    # Logging
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save config
    with open(f"{output_dir}/config.json", "w") as f:
        print(f"Config saved in {output_dir}/config.json.")
        config_dict = args.__dict__.copy()
        config_dict['enable_depth'] = args.enable_depth  # NEW
        config_dict['depth_loss_weight'] = args.depth_loss_weight  # NEW
        json.dump(config_dict, f)

    # Save output of terminal in log file
    sys.stdout = Logger(os.path.join(output_dir, "log_train.txt"))
    arguments = str(args).split(", ")
    print("=========================\nConfigs:{}\n=========================")
    for i in range(len(arguments)):
        print(arguments[i])
    print(
        "Hyperparameters from config file: "
        + ", ".join(f"{k}={v}" for k, v in config_.items())
    )
    print("=========================")

    ########## Reproducibility ##########

    set_seed(config.training["seed"])

    ########## Build training set ##########

    dataset = build_dataset(
        root_dir=args.dataset_dir,
        dataset_name=config.training["dataset"],
        dataset_set=config.training["dataset_set"],
        config=config,
        for_eval=False,
    )

    dataset_set = config.training["dataset_set"]
    str_set = dataset_set if dataset_set is not None else ""
    print(f"\nBuilding dataset {dataset.name}{str_set} of {len(dataset)}")

    ########## Define Peekaboo ##########

    model = PeekabooModel(
        vit_model=config.model["pre_training"],
        vit_arch=config.model["arch"],
        vit_patch_size=config.model["patch_size"],
        enc_type_feats=config.peekaboo["feats"],
        enable_depth=args.enable_depth,  # NEW: Pass depth flag to model
    )

    ########## Training and evaluation ##########

    print(f"\nStarted training on {dataset.name} [tensorboard dir: {output_dir}]")
    model = train_model(
        model=model,
        config=config,
        dataset=dataset,
        dataset_dir=args.dataset_dir,
        tensorboard_log_dir=output_dir,
        visualize_freq=args.visualization_freq,
        save_model_freq=args.save_model_freq,
        enable_depth=args.enable_depth,  # NEW
        depth_loss_weight=args.depth_loss_weight,  # NEW
    )
    print(f"\nTraining done, Peekaboo model saved in {output_dir}.")


if __name__ == "__main__":
    main()