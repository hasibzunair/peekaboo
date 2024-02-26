# Copyright 2022 - Valeo Comfort and Driving Assistance - Oriane Siméoni @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import random
import argparse
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from model import FoundModel
from evaluation.saliency import evaluate_saliency
from misc import (
    batch_apply_bilateral_solver,
    set_seed,
    load_config,
)

from datasets.datasets import build_dataset

def train_model(
    model,
    config,
    dataset,
    dataset_dir,
    visualize_freq=10,
    save_model_freq=500,
    tensorboard_log_dir=None,
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

    # Deconvolution
    sigmoid = nn.Sigmoid()
    model.decoder.train()
    model.decoder.to("cuda")

    # Optimization
    criterion = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    optimizer = torch.optim.AdamW(
                                  model.decoder.parameters(),
                                  lr=config.training["lr0"]
                                 )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.training["step_lr_size"],
        gamma=config.training["step_lr_gamma"],
    )

    # Dataset   
    trainloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=config.training["batch_size"], 
                                            shuffle=True, 
                                            num_workers=2
                                            )

    n_iter = 0
    for epoch in range(config.training["nb_epochs"]):
        running_loss = 0.0
        tbar = tqdm(enumerate(trainloader, 0), leave=None)
        for i, data in tbar:

            # get the inputs
            inputs, masked_inputs, scribbles, input_nonorm, masked_input_nonorm, gt_labels, img_paths = data
            
            #######
            # For debug
            # def to_img(ten):
            #     #ten =(input_nonorm[0].permute(1,2,0).detach().cpu().numpy()+1)/2
            #     ten =(ten.permute(1,2,0).detach().cpu().numpy())
            #     ten=(ten*255).astype(np.uint8)
            #     #ten=cv2.cvtColor(ten,cv2.COLOR_RGB2BGR)
            #     return ten
            # import pdb; pdb.set_trace()
            # im = to_img(input_nonorm[0])
            # plt.imshow(im); plt.show()
            #######

            # inputs and gt labels
            inputs = inputs.to("cuda")
            masked_inputs = masked_inputs.to("cuda")
            gt_labels = gt_labels.to("cuda")

            # zero the parameter gradients
            optimizer.zero_grad()

            #### Forward step ####
            # Goal: General idea is that the model learns to predict a 
            # smoothed version of the complemnent of the coarse bkg masks and its 
            # own prediction, so that it quickly converges to refined masks.
            # M_s
            preds, _, shape_f, att = model.forward_step(inputs)

            # Binarization
            preds_mask = (sigmoid(preds.detach()) > 0.5).float()

            # Apply bilateral solver
            # M_s_hat
            preds_mask_bs, _ = batch_apply_bilateral_solver(
                                    data,
                                    preds_mask.detach()
                                )
            
            # flat_preds, flattened from model preds (39200, 1)
            flat_preds = preds.permute(0, 2, 3, 1).reshape(-1, 1)

            #### Compute loss (L_s = (M_s, M_s_hat))  to smooth and refine predictions ####
            # Goal: Predict a refined version of the prediction itself 
            # and force quality of mask edges.
            alpha = config.training["w_bs_loss"]
            preds_bs_loss = alpha * criterion(
                flat_preds, preds_mask_bs.reshape(-1).float()[:,None]
            )
            print(preds_bs_loss)
            writer.add_scalar("Loss/L_s", preds_bs_loss, n_iter) # self_bs
            loss = preds_bs_loss


            ###### Masked Supervised Learning ######
            # Context Branch
            preds_cb, _, shape_f_cb, att_cb = model.forward_step(masked_inputs)
            preds_mask_cb = (sigmoid(preds_cb.detach()) > 0.5).float()
            preds_mask_cb_bs, _ = batch_apply_bilateral_solver(
                                     data,
                                     preds_mask_cb.detach()
                                 )
            flat_preds_cb = preds_cb.permute(0, 2, 3, 1).reshape(-1, 1)

            # Context branch loss
            beta = 1
            preds_bs_cb_loss = beta * criterion(
                 flat_preds_cb, preds_mask_cb_bs.reshape(-1).float()[:,None]
                 )
            writer.add_scalar("Loss/L_context", preds_bs_cb_loss, n_iter)
            #loss += preds_bs_cb_loss

            # Task Similarity loss
            gamma = 1
            # task_sim_loss = gamma *  criterion(
            #      flat_preds, flat_preds_cb
            #      )
            task_sim_loss = gamma *  criterion_mse(
                 preds_mask_bs.reshape(-1).float()[:,None], preds_mask_cb_bs.reshape(-1).float()[:,None]
                 )
            writer.add_scalar("Loss/L_tasksim", task_sim_loss, n_iter)
            #loss += task_sim_loss
            ###### End of Masked Supervised Learning ######


            if n_iter < config.training["stop_bkg_loss"]:
                # Get pseudo_labels used as gt
                # Refined (M_f)
                masks, _ = model.get_bkg_pseudo_labels_batch(
                            att=att,
                            shape_f=shape_f,
                            data=data,
                            shape=preds.shape[-2:],
                        )
                flat_labels = masks.reshape(-1)

                #### Compute loss L_f = (M_s, Refined (M_f)) to guide predictions towards background masks ####
                # Goal: Initialize and guide to predict the compliment M_f of the coarse 
                # bkg mask M_b refined by bilateral solver
                bkg_loss = criterion(
                    flat_preds, flat_labels.float()[:, None]
                )
                writer.add_scalar("Loss/L_f", bkg_loss, n_iter)
                loss += bkg_loss
            
            # Add regularization when bkg loss stopped
            else:
                
                ### Compute loss betn soft masks and their binarized versions ####
                self_loss = criterion(
                            flat_preds, preds_mask.reshape(-1).float()[:,None]
                        )

                self_loss =  self_loss * config.training["w_self_loss"]
                loss += self_loss
                writer.add_scalar("Loss/L_regularization", self_loss, n_iter)
            
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
                mp_grid = torchvision.utils.make_grid(preds_mask_cb[:5])
                writer.add_image("training/masked_preds", mp_grid, n_iter)
                
                # Visualize masks
                # if n_iter < config.training["stop_bkg_loss"]:
                #     p_grid = torchvision.utils.make_grid(masks[:5].unsqueeze(1))
                #     writer.add_image("training/bkg_masks", p_grid, n_iter)

            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/total_loss", loss, n_iter)
            writer.add_scalar("params/lr", optimizer.param_groups[0]["lr"], n_iter)
            scheduler.step()

            # Statistics
            running_loss += loss.item()
            tbar.set_description(
                f"{dataset.name}| train | iter {n_iter} | loss: ({running_loss / (i + 1):.3f}) "
            )

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
                        val_dataset,
                        model=model,
                        n_iter=n_iter,
                        writer=writer
                    )
        
            if n_iter == config.training["max_iter"]:
                model.decoder_save_weights(save_dir, n_iter)
                print("\n----"
                      "\nTraining done.")
                writer.close()
                return model

            n_iter += 1
    
    # Save model
    model.decoder_save_weights(save_dir, n_iter)

    writer.close()
    return model


if __name__ == "__main__":

    ########## Reproducibility ##########
    # random.seed(0)
    # np.random.seed(0)
    # os.environ["PYTHONHASHSEED"] = str(0)
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)
    # torch.cuda.manual_seed_all(0)
    # if torch.cuda.is_available():
    #     torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    ########## Get arguments ##########
    parser = argparse.ArgumentParser(
                description = 'Training of MSL',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Exp name."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="outputs",
        help="Logging and output directory."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Root directories of training and evaluation datasets."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/msl_DUTS-TR.yaml",
        help="Path of config file."
    )
    parser.add_argument(
        "--save-model-freq",
        type=int,
        default=250,
        help="Frequency of model saving."
    )
    parser.add_argument(
        "--visualization-freq",
        type=int,
        default=10,
        help="Frequency of prediction visualization in tensorboard."
    )
    

    args = parser.parse_args()
    print(args.__dict__)

    # Configuration
    config = load_config(args.config)

    # Exp name
    exp_name = "MSL-{}-{}{}".format(
                                    config.training["dataset"],
                                    config.model["arch"],
                                    config.model["patch_size"]
                                )

    if args.exp_name is not None:
        exp_name = f"{args.exp_name}-{exp_name}"
        
    # Log dir
    output_dir = os.path.join(
                    args.log_dir,
                    exp_name
                 )
    # Logging
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save config
    with open(f'{output_dir}/config.json', 'w') as f:
        print(f"Config saved in {output_dir}/config.json.")
        json.dump(args.__dict__, f)
    
    # ------------------------------------
    # Set seed
    set_seed(config.training["seed"])

    # ------------------------------------
    # Build the training set
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

    # ------------------------------------
    # Define the model
    model = FoundModel(
                vit_model=config.model["pre_training"],
                vit_arch=config.model["arch"],
                vit_patch_size=config.model["patch_size"],
                enc_type_feats=config.found["feats"],
                bkg_type_feats=config.found["feats"],
                bkg_th=config.found["bkg_th"]
            )

    # ------------------------------------
    # Training
    print(f"\nStarted training on {dataset.name} [tensorboard dir: {output_dir}]")
    model = train_model(
                model=model,
                config=config,
                dataset=dataset,
                dataset_dir=args.dataset_dir,
                tensorboard_log_dir=output_dir,
                visualize_freq=args.visualization_freq,
                save_model_freq=args.save_model_freq,
            )
    print(f"\nTraining done, MSL model saved in {output_dir}.")