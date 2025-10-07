# This code is based on Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation 
# Copyright (c) 2023, Zikang Zhou. 
# Modifications Copyright (c)  Da Saem Lee, 2025

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from datamodules import ArgoverseV2DataModule
from predictors import  PDInit, PDTraj
from transforms import TargetBuilderTraj, TargetBuilderInit

if __name__ == '__main__':
    pl.seed_everything(2024, workers=True)
    
    
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=str, default="1")
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    
    parser.add_argument('--guid_sampling', choices=['no_guid', 'guid'],default = 'no_guid')
    parser.add_argument('--guid_task', choices=['none', 'goal', 'target_vel', 'target_vego','rand_goal_rand','rand_goal_rand_o'],default = 'none')
    parser.add_argument('--guid_method', choices=['none', 'ECM', 'ECMR'],default = 'none')
    parser.add_argument('--guid_plot',choices=['no_plot', 'plot'],default = 'no_plot')
    parser.add_argument('--plot',choices=['no_plot', 'plot'],default = 'no_plot')
    parser.add_argument('--path_pca_V_k', type = str,default = 'none')

    parser.add_argument('--cond_norm', type = int, default = 0)
    parser.add_argument('--cost_param_costl', type = float, default = 1.0)
    parser.add_argument('--cost_param_threl', type = float, default = 1.0)
    
    
    parser.add_argument('--stage', type = str, default = 'init', choices = ['init', 'traj'])
    
    
    PDTraj.add_model_specific_args(parser)
    
    args = parser.parse_args()
    if args.stage == 'init':
        args = parser.parse_args()
        model = PDInit(args)
        model_checkpoint = ModelCheckpoint(monitor='val_trans_loss', save_top_k=5, mode='min')
        args.train_transform = TargetBuilderInit(50, 60)
        args.val_transform = TargetBuilderInit(50, 60)

    elif args.stage == 'traj':
        args = parser.parse_args()
        model = PDTraj(args)
        
        model_checkpoint = ModelCheckpoint(monitor='val_minADE', save_top_k=5, mode='min')
        args.train_transform = TargetBuilderTraj(50, 60)
        args.val_transform = TargetBuilderTraj(50, 60)
    else:
        raise NotImplementedError
    
    model.add_extra_param(args)
    
    datamodule = {
        'argoverse_v2': ArgoverseV2DataModule,
    }[args.dataset](**vars(args))
    
    

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accelerator=args.accelerator, 
                         devices=args.devices, 
                         strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True),
                         callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs, 
                         check_val_every_n_epoch=args.check_val_every_n_epoch,

                         num_sanity_val_steps = 1)
    trainer.fit(model, datamodule)