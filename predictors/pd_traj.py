# This code is based on Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation, 
# Copyright (c) 2023, Zikang Zhou. All rights reserved.
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
from pathlib import Path
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from metrics import MR
from metrics import minADE
from metrics import minFDE
from metrics import CollisionTraj, OffRoadTraj
from modules import TrajDiffusion, QCNetMapEncoder

import numpy as np
from pathlib import Path

try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object

from av2.datasets.motion_forecasting import scenario_serialization
from visualization import *
from av2.map.map_api import ArgoverseStaticMap
import os

import gc
class PDTraj(pl.LightningModule):

    def __init__(self,
                 args,
                 **kwargs) -> None:
        super(PDTraj, self).__init__()
        self.save_hyperparameters()
        self.dataset = args.dataset
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.output_head = args.output_head
        self.init_timestep = args.init_timestep
        self.num_generation_timestep = args.num_generation_timestep
        self.num_freq_bands = args.num_freq_bands
        self.num_map_layers = args.num_map_layers
        self.num_heads = args.num_heads
        self.head_dim = args.head_dim
        self.dropout = args.dropout
        self.pl2pl_radius = args.pl2pl_radius
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        # self.T_max = args.T_max
        self.submission_dir = args.submission_dir
        self.submission_file_name = args.submission_file_name
        self.diff_type = args.diff_type
        self.sampling = args.sampling
        self.sampling_stride = args.sampling_stride
        self.num_diffusion_steps = args.num_diffusion_steps
        self.num_eval_samples = args.num_eval_samples
        self.path_pca_s_mean = args.path_pca_s_mean
        self.path_pca_VT_k = args.path_pca_VT_k
        self.path_pca_latent_mean = args.path_pca_latent_mean
        self.path_pca_latent_std = args.path_pca_latent_std
        self.s_mean = None
        self.VT_k = None
        self.latent_mean = None
        self.latent_std = None
        self.m_dim = args.m_dim
        self.root = args.root
        
        self.check_param()

        
        self.qcnet_mapencoder = QCNetMapEncoder(dataset=args.dataset,
                                                input_dim=self.input_dim,
                                                hidden_dim=self.hidden_dim,
                                                pl2pl_radius=self.pl2pl_radius,
                                                num_freq_bands=self.num_freq_bands,
                                                num_layers=self.num_map_layers,
                                                num_heads=self.num_heads,
                                                head_dim=self.head_dim,
                                                dropout=self.dropout)
        

        map_encoder_load = torch.load(args.qcnet_map_ckpt_path)
        self.qcnet_mapencoder.load_state_dict(map_encoder_load)
        
        # freeze 
        for params in self.qcnet_mapencoder.parameters():
            params.requires_grad = False
        
        
        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.linear = nn.Linear(10,2)
        
        self.joint_diffusion = TrajDiffusion(args=args)
        
        self.minADE = minADE(max_guesses=1)
        self.minFDE = minFDE(max_guesses=6)
        self.MR = MR(max_guesses=1)
        
        self.CollisionTraj = CollisionTraj()
        self.CollisionTraj_gt = CollisionTraj()
        
        self.OffRoadTraj = OffRoadTraj()
        self.OffRoadTraj_gt = OffRoadTraj()
        
        
    def add_extra_param(self, args):
        self.guid_sampling = args.guid_sampling
        self.guid_task = args.guid_task
        self.guid_method = args.guid_method
        self.guid_plot = args.guid_plot
        self.plot = args.plot
        self.path_pca_V_k = args.path_pca_V_k
        self.V_k = None
        self.root = args.root
        self.cond_norm = args.cond_norm
        self.cost_param_costl = args.cost_param_costl
        self.cost_param_threl = args.cost_param_threl
        if args.__contains__('ckpt_path'):
            self.ckpt_path = args.ckpt_path
        else:
            self.ckpt_path = None
        
        
    def check_param(self):
        if self.sampling == 'ddpm':
            self.sampling_stride = 1
        elif self.sampling == 'ddim':
            self.sampling_stride = int(self.sampling_stride)
            if self.sampling_stride > self.num_diffusion_steps - 1:
                print('ddim stride > diffusion steps.')
                exit()
            scale = self.num_diffusion_steps / self.sampling_stride
            if abs(scale - int(scale)) > 0.00001:
                print('mod(diffusion steps, ddim stride) != 0')
                exit()

    def forward(self, data: HeteroData):
        scene_enc = self.qcnet.encoder(data)
        x = torch.ones(32,10).to(scene_enc['x_a'].device)
        return self.linear(x)

    def normalize(self, original_data, mean, std):
        if original_data.dim() == 2:
            if mean.dim() == 1:
                return (original_data - mean.unsqueeze(0))/(std.unsqueeze(0)+0.1)
            if mean.dim() == 2:
                return (original_data - mean)/(std+0.1)
        elif original_data.dim() == 3:
            if mean.dim() == 1:
                return (original_data - mean.unsqueeze(0).unsqueeze(0))/(std.unsqueeze(0).unsqueeze(0)+0.1)
            if mean.dim() == 2:
                return (original_data - mean.unsqueeze(1))/(std.unsqueeze(1)+0.1)
        else:
            raise ValueError('normalized data should 2-dimensional or 3-dimensional.')
    
    def unnormalize(self, original_data, mean, std):
        if original_data.dim() == 2:
            if mean.dim() == 1:
                return original_data*(std.unsqueeze(0)+0.1) + mean.unsqueeze(0)
            if mean.dim() == 2:
                return original_data*(std+0.1) + mean
        elif original_data.dim() == 3:
            if mean.dim() == 1:
                return original_data * (std.unsqueeze(0).unsqueeze(0)+0.1) + mean.unsqueeze(0).unsqueeze(0)
            if mean.dim() == 2:
                return original_data * (std.unsqueeze(1)+0.1) + mean.unsqueeze(1)
        else:
            raise ValueError('normalized data should 2-dimensional or 3-dimensional.')
    
    def create_rot_mat(self, theta, num_agents):
        if theta.dim() == 1:
            cos, sin = theta.cos(), theta.sin()
            mat = torch.zeros(num_agents, 2, 2, device=theta.device)
            mat[:, 0, 0] = cos
            mat[:, 0, 1] = -sin
            mat[:, 1, 0] = sin
            mat[:, 1, 1] = cos
            
        elif theta.dim() == 2:
            num_agents, num_samples = theta.shape
            cos, sin = theta.cos(), theta.sin()
            mat = torch.zeros(num_agents, num_samples, 2, 2, device=theta.device)
            mat[:, :, 0, 0] = cos
            mat[:, :, 0, 1] = -sin
            mat[:, :, 1, 0] = sin
            mat[:, :, 1, 1] = cos
        return mat
    
    
    def interpolate_data(self, data_seq, reg_mask,num_agent):
        _, seq_len = reg_mask.shape  # Assuming seq_len = 60 as per the code
        reg_start_list = torch.zeros((num_agent, seq_len), dtype=torch.int64).to(data_seq.device)
        reg_end_list = torch.zeros((num_agent, seq_len), dtype=torch.int64).to(data_seq.device)

        # Find start and end indices for True-to-False and False-to-True transitions
        start_indices = ((reg_mask[:, :-1] == True) & (reg_mask[:, 1:] == False)).nonzero(as_tuple=True)
        end_indices = ((reg_mask[:, :-1] == False) & (reg_mask[:, 1:] == True)).nonzero(as_tuple=True)

        # Populate start and end lists
        reg_start_list[start_indices[0], start_indices[1]] = start_indices[1]
        reg_end_list[end_indices[0], end_indices[1]] = end_indices[1] + 1

        # Create indices for all positions in data_seq
        j_indices = torch.arange(seq_len - 1, device=data_seq.device).expand(num_agent, -1)

        # Identify positions where interpolation is needed (reg_mask is False)
        interpolation_mask = ~reg_mask[:, :-1]  # Negate to get False positions
        
        if len(torch.where(interpolation_mask == True)[0]) == 0:
            return data_seq
        # Use reg_start_list and reg_end_list to gather indices where interpolation starts and ends
        start_ids = reg_start_list.gather(1, interpolation_mask.long().cumsum(dim=1))
        end_ids = reg_end_list.gather(1, interpolation_mask.long().cumsum(dim=1))

        # Get start and end points for interpolation using gathered indices
        start_pts = data_seq.gather(1, start_ids.unsqueeze(-1).expand(-1, -1, 3))
        end_pts = data_seq.gather(1, end_ids.unsqueeze(-1).expand(-1, -1, 3))

        # Calculate interpolated values for False positions in reg_mask
        interpolation_values = start_pts + (end_pts - start_pts) / (end_ids - start_ids).unsqueeze(-1) * (j_indices.unsqueeze(-1) - start_ids.unsqueeze(-1))

        # Update data_seq where interpolation is required
        data_seq[interpolation_mask, :-1] = interpolation_values[interpolation_mask]

        return data_seq
                        
    def training_step(self,
                      data,
                      batch_idx):
        
        print_flag = False
        if batch_idx % 100 == 0:
            print_flag = True
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        
    
        scene_enc = self.qcnet_mapencoder(data)
        
        cand_enc = {}
        
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)

        reg_mask = data['agent']['predict_mask'][:, self.init_timestep:]
        
        eval_mask = data['agent']['mask']
        
        gt = gt[eval_mask][..., :self.output_dim]
        

        reg_mask = reg_mask[eval_mask]
        num_agent = gt.size(0)
        gt = self.interpolate_data(gt, reg_mask, num_agent)
        
        flat_gt = gt.reshape(gt.size(0), -1)
        device = gt.device
        if self.s_mean == None:
            self.load_vars(device)
        
        target_mode = torch.matmul(flat_gt-self.s_mean, self.VT_k)
        target_mode = self.normalize(target_mode, self.latent_mean, self.latent_std)

        rec_flat_gt = gt.unsqueeze(1)
        
        # no need for vd, but for others it need. Not removing it for now
        
        
        num_samples = 1
        
        # PCA on candidates
        primitives = data['agent']['rotated_candidate_list'].reshape(-1, self.num_generation_timestep*2)
        primitives = torch.matmul(primitives-self.s_mean, self.VT_k)
        primitives = self.normalize(primitives, self.latent_mean, self.latent_std)
        cand_enc['primitives'] = primitives
        
        
        loss, x_0_reconstructed_latent = self.joint_diffusion.get_loss(target_mode, data = data, scene_enc = scene_enc, eval_mask=eval_mask, clean_data=cand_enc, num_samples=num_samples)
        
        x_0_reconstructed = self.unnormalize(x_0_reconstructed_latent, self.latent_mean, self.latent_std)
        x_0_reconstructed = torch.matmul(x_0_reconstructed, self.V_k) + self.s_mean
        x_0_reconstructed = x_0_reconstructed.view(-1, num_samples, 60, 2)
 
        
        all_rel_origin = data['agent']['position'][eval_mask, self.init_timestep].unsqueeze(1).repeat(1, num_samples, 1)[..., :2]
        all_rel_theta = data['agent']['heading'][eval_mask, self.init_timestep].unsqueeze(1).repeat(1, num_samples)
        all_rel_rot_mat = self.create_rot_mat(all_rel_theta, all_rel_theta.shape[0])
        all_rel_rot_mat_inv = all_rel_rot_mat.permute(0, 1, 3, 2)

        rec_flat_gt = rec_flat_gt.repeat(1, num_samples, 1, 1)
        rec_traj_world_gt = torch.matmul(rec_flat_gt[:, :, :, :2],
                                all_rel_rot_mat_inv) + all_rel_origin[:, :, :2].view(-1, num_samples, 1, 2) 
        
        rec_traj_world_pred = torch.matmul(x_0_reconstructed[:, :, :, :2],
                                all_rel_rot_mat_inv) + all_rel_origin[ :, :2].view(-1, num_samples, 1, 2) 
        
        traj_loss = torch.nn.HuberLoss(reduction='none')(rec_traj_world_pred, rec_traj_world_gt).squeeze(1).sum(dim=-1).sum(dim=-1).mean()
        self.log('train/diff_loss', loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_lr', self.optimizers().param_groups[0]['lr'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_traj_loss', traj_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        
        if print_flag:
            
            print(batch_idx, ", diff_loss:" , loss.item(), ", traj_loss:" , traj_loss.item())
            
        return loss

    def validation_step(self,
                    data,
                    batch_idx):
        if self.guid_sampling == 'no_guid':
            self.validation_step_norm(data, batch_idx)
        elif self.guid_sampling == 'guid':
            self.validation_step_guid(data, batch_idx)
    def on_validation_end(self):
        torch.cuda.synchronize()  # if on GPU
        
        gc.collect()
    def on_validation_end(self):
        torch.cuda.synchronize()  # if on GPU
        
        gc.collect()
    def load_vars(self, device):
        s_mean = np.load(self.path_pca_s_mean)
        self.s_mean = torch.tensor(s_mean).to(device)
        VT_k = np.load(self.path_pca_VT_k)
        
        self.VT_k = torch.tensor(VT_k).to(device)
        if self.path_pca_V_k != 'none':
            V_k = np.load(self.path_pca_V_k)
            self.V_k = torch.tensor(V_k).to(device)
        else:
            self.V_k = self.VT_k.transpose(0,1)
        latent_mean = np.load(self.path_pca_latent_mean)
        self.latent_mean = torch.tensor(latent_mean).to(device)
        latent_std = np.load(self.path_pca_latent_std) * 2
        self.latent_std = torch.tensor(latent_std).to(device)
        
    def validation_step_norm(self,
                    data,
                    batch_idx):
        print_flag = False
        if batch_idx % 100 == 0:
            print_flag = True
            
        
        data_batch = batch_idx
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        
        reg_mask = data['agent']['predict_mask'][:, self.init_timestep:]
        scene_enc = self.qcnet_mapencoder(data)


        cand_enc = {}
    
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        
        if self.s_mean == None:
            self.load_vars(self.device)
        
        mask = data['agent']['mask']
        
        gt_n = gt[mask][..., :self.output_dim]
        
        # scale
        gt_n[0,:,:] = (gt_n[0,:,:] - gt_n[0,0:1,:]) / 4 * 3 + gt_n[0,0:1,:]
        
        
        flat_gt = gt_n.reshape(gt_n.size(0),-1)
        
        k_vector = torch.matmul(flat_gt-self.s_mean, self.VT_k)
        rec_flat_gt = torch.matmul(k_vector, self.V_k) + self.s_mean
        rec_gt = rec_flat_gt.view(-1,60,2)
        
        target_mode = torch.matmul(flat_gt-self.s_mean, self.VT_k)
        target_mode = self.normalize(target_mode, self.latent_mean, self.latent_std)

        if self.dataset == 'argoverse_v2':
            eval_mask =  data['agent']['mask']
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        valid_mask_eval = reg_mask[eval_mask]

        gt_eval = gt[eval_mask]
    
 
        self.joint_diffusion.eval()
        num_samples = self.num_eval_samples

        if_output_diffusion_process = False
        
        primitives = data['agent']['rotated_candidate_list'].reshape(-1, self.num_generation_timestep*2)
        
        primitives = torch.matmul(primitives-self.s_mean, self.VT_k)
        primitives = self.normalize(primitives, self.latent_mean, self.latent_std)
        cand_enc['primitives'] = primitives
        
        
        if if_output_diffusion_process:
            reverse_steps = self.num_diffusion_steps
            pred_modes = self.joint_diffusion.sample(num_samples, data = data, scene_enc = scene_enc, 
                                                    sampling=self.sampling,
                                                    stride=self.sampling_stride,eval_mask=eval_mask,
                                                    if_output_diffusion_process=if_output_diffusion_process,
                                                    reverse_steps=reverse_steps, clean_data=primitives)

            
            inter_latents = pred_modes[::1]
            inter_trajs = []
            for latent in inter_latents:
                unnorm_pred_modes = self.unnormalize(latent,self.latent_mean, self.latent_std)
                rec_traj = torch.matmul(unnorm_pred_modes.permute(1,0,2), (self.V_k).unsqueeze(0).repeat(self.num_eval_samples,1,1)) + self.s_mean.unsqueeze(0)
                rec_traj = rec_traj.permute(1,0,2)
                rec_traj = rec_traj.view(rec_traj.size(0), rec_traj.size(1),self.num_generation_timestep,2)
                inter_trajs.append(rec_traj)
            
            
            pred_modes = pred_modes[-1]
            
                
            
        else:
            
            start_data = None
            reverse_steps = None
            pred_modes = self.joint_diffusion.sample(num_samples, data = data, scene_enc = scene_enc, 
                                                    sampling=self.sampling,
                                                    stride=self.sampling_stride,eval_mask=eval_mask,
                                                    if_output_diffusion_process=if_output_diffusion_process,
                                                    reverse_steps=reverse_steps, clean_data=cand_enc)

        
        unnorm_pred_modes = self.unnormalize(pred_modes,self.latent_mean, self.latent_std)
        rec_traj = torch.matmul(unnorm_pred_modes.permute(1,0,2), (self.V_k).unsqueeze(0).repeat(self.num_eval_samples,1,1)) + self.s_mean.unsqueeze(0)
        rec_traj = rec_traj.permute(1,0,2)
        rec_traj = rec_traj.view(rec_traj.size(0), rec_traj.size(1),self.num_generation_timestep,2)


        random_modes = torch.fmod(torch.randn_like(pred_modes),3) / 2
        unnorm_pred_modes = self.unnormalize(random_modes,self.latent_mean, self.latent_std)
        random_traj = torch.matmul(unnorm_pred_modes.permute(1,0,2), (self.V_k).unsqueeze(0).repeat(self.num_eval_samples,1,1)) + self.s_mean.unsqueeze(0)
        random_traj = random_traj.permute(1,0,2)
        random_traj = random_traj.view(random_traj.size(0), random_traj.size(1),self.num_generation_timestep,2)
        
        if True in torch.isnan(pred_modes):
            print('nan')
            print(data_batch)
            exit()

        batch_idx = data['agent']['batch'][eval_mask]
        num_scenes = batch_idx[-1].item()+1
        num_agents_per_scene = rec_traj.new_tensor([(batch_idx == i).sum() for i in range(num_scenes)]).type(torch.int64)

        all_rel_origin_eval = data['agent']['position'][eval_mask, self.init_timestep, :2]
        all_rel_theta_eval = data['agent']['heading'][eval_mask, self.init_timestep]
        all_rel_rot_mat = self.create_rot_mat(all_rel_theta_eval, all_rel_theta_eval.shape[0])
        all_rel_rot_mat_inv = all_rel_rot_mat.permute(0, 2, 1)


        scenario_id = data['scenario_id']
        if os.path.exists(f'starting_hmp/{scenario_id[0]}.pt'):
            initials = torch.load(f'starting_hmp/{scenario_id[0]}.pt')
            init_pos_all = initials[..., :2]
            init_heading_vector_all = initials[..., 2:4]
            
            init_heading_all = torch.atan2(init_heading_vector_all[..., 1], init_heading_vector_all[..., 0])
        else:
            init_pos_all = data['agent']['position'][eval_mask][:, self.init_timestep, :2]
            init_heading_all = data['agent']['heading'][eval_mask][:, self.init_timestep]
            init_heading_vector_all = torch.stack([init_heading_all.cos(), init_heading_all.sin()], dim=-1)
            
            init_velocity = data['agent']['velocity'][eval_mask][:, self.init_timestep, :]

        pred_rot_mat_inv = self.create_rot_mat(init_heading_all, init_heading_all.shape[0]).permute(0, 2, 1)
        
        rec_traj_world = torch.matmul(rec_traj[:, :, :, :2],
                                pred_rot_mat_inv.unsqueeze(1)) + init_pos_all[:, :2].reshape(-1, 1, 1, 2)
            
        gt_eval_world = torch.matmul(gt_eval[:, :, :2],
                                all_rel_rot_mat_inv) + all_rel_origin_eval[:, :2].reshape(-1, 1, 2)
        
            
        self.minADE.update(pred=rec_traj_world, target=gt_eval_world[..., :self.output_dim],
                        valid_mask=valid_mask_eval)
        self.log('val/minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)

        batch_agent_idx = data['agent']['batch'][eval_mask]
        self.CollisionTraj.update(pred=rec_traj_world, agent_batch=batch_agent_idx)
        self.log('val/CollisionTraj', self.CollisionTraj, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
        
        self.CollisionTraj_gt.update(pred=gt_eval_world[..., :self.output_dim], agent_batch=batch_agent_idx)
        self.log('val/CollisionTraj_gt', self.CollisionTraj_gt, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
        map_pts = data['map_point']
        self.OffRoadTraj.update(pred=rec_traj_world, agent_batch=batch_agent_idx, map_pts=map_pts)
        self.log('val/OffRoadTraj', self.OffRoadTraj, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
        
        self.OffRoadTraj_gt.update(pred=gt_eval_world[..., :self.output_dim], agent_batch=batch_agent_idx, map_pts=map_pts)
        
        self.log('val/OffRoadTraj_gt', self.OffRoadTraj_gt, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
        
        self.MR.update(pred=rec_traj_world, target=gt_eval_world[..., :self.output_dim],
                        valid_mask=valid_mask_eval)
        self.log('val/MR', self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
               
        batch_agent_idx = data['agent']['batch'][eval_mask]


        if print_flag:
            print('truncted minADE',self.minADE.compute())
            print('truncted MR',self.MR.compute())
        goal_point = gt_eval[:,-1,:2]
        plot = self.plot
        if plot == 'plot':
            # gt_eval: relative position, rotated to its origin, eval_masked
            if if_output_diffusion_process:
                inter_trajs_world = []
                rot_mat = all_rel_rot_mat_inv
                origin_eval = all_rel_origin_eval
                for traj in inter_trajs:
                    traj_world = torch.matmul(traj[:, :, :, :2],
                                    rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
                    inter_trajs_world.append(traj_world.detach().cpu().numpy())

            gt_eval_world = torch.matmul(gt_eval[:, :, :2],
                                    all_rel_rot_mat_inv) + all_rel_origin_eval[:, :2].reshape(-1, 1, 2)
            
            gt_eval_world = gt_eval_world.detach().cpu().numpy()
            
            goal_point_world = torch.matmul(goal_point[:, None, None, :],
                                            all_rel_rot_mat_inv.unsqueeze(1)) + all_rel_origin_eval[:, :2].reshape(-1, 1, 1, 2)
            goal_point_world = goal_point_world.squeeze(1).squeeze(1)
            goal_point_world = goal_point_world.detach().cpu().numpy()


            img_folder = 'visual'
            if self.ckpt_path:
                sub_folder = self.ckpt_path.split('/')[-3]
            else:
                sub_folder = 'a'
            rec_traj_world = rec_traj_world.detach().cpu().numpy()
            for i in range(num_scenes):
                start_id = torch.sum(num_agents_per_scene[:i])
                end_id = torch.sum(num_agents_per_scene[:i+1])
                
                if end_id - start_id == 1:
                    continue
                
                temp = gt_eval[start_id:end_id]
                temp_start = temp[:,0,:]
                temp_end = temp[:,-1,:]
                norm = torch.norm(temp_end-temp_start,dim=-1)
                if torch.max(norm) < 10:
                    continue
            
                scenario_id = data['scenario_id'][i]
                base_path_to_data = Path(f'{self.root}/val/raw')
                scenario_folder = base_path_to_data / scenario_id
                
                static_map_path = scenario_folder / f"log_map_archive_{scenario_id}.json"
                scenario_path = scenario_folder / f"scenario_{scenario_id}.parquet"


                scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
                static_map = ArgoverseStaticMap.from_json(static_map_path)
                
                viz_output_dir = Path(img_folder) / sub_folder
                os.makedirs(viz_output_dir,exist_ok=True)

                viz_save_path = viz_output_dir / ('b'+ str(data_batch)+'_s'+str(i)+'_'+self.sampling+'.svg')
                
                additional_traj = {}
                additional_traj['gt'] = gt_eval_world[start_id:end_id]
                additional_traj['rec_traj'] = rec_traj_world[start_id:end_id]
                
                traj_visible = {}
                                
                traj_visible['gt'] = False
                traj_visible['gt_goal'] = False
                traj_visible['goal_point'] = False
                traj_visible['marg_traj'] = False
                traj_visible['rec_traj'] = True
                                
                visualize_scenario_prediction(scenario, static_map, additional_traj, traj_visible, viz_save_path, data)
                traj_visible['gt'] = True
                traj_visible['marg_traj'] = False
                if if_output_diffusion_process:
                    for j in range(len(inter_trajs_world)):
                        traj = inter_trajs_world[j]
                        additional_traj['rec_traj'] = traj[start_id:end_id]
                        viz_save_path = viz_output_dir / ('b'+ str(data_batch)+'_s'+str(i)+'_'+self.sampling+'_'+'inter_'+str(j)+'_'+'reverse_'+str(reverse_steps)+'.jpg')
                        visualize_scenario_prediction(scenario, static_map, additional_traj, traj_visible, viz_save_path)



    def validation_step_guid(self,
                        data,
                        batch_idx):
        print_flag = False
        if batch_idx % 1 == 0:
            print_flag = True
        
        data_batch = batch_idx
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        
        reg_mask = data['agent']['predict_mask'][:, self.init_timestep:]
        
        scene_enc = self.qcnet.encoder.map_encoder(data)
        
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        
        
        if self.s_mean == None:
            self.load_vars(gt.device)
            
        
        eval_mask = data['agent']['category'] >= 2
        
        mask = (data['agent']['category'] >= 2) & (reg_mask[:,-1] == True) & (reg_mask[:,0] == True)
        gt_n = gt[mask][..., :self.output_dim]
        reg_mask_n = reg_mask[mask]
        num_agent = gt_n.size(0)
        gt_n = self.interpolate_data(gt_n, reg_mask_n, num_agent)
        
        
        flat_gt = gt_n.reshape(gt_n.size(0),-1)
        k_vector = torch.matmul(flat_gt-self.s_mean, self.VT_k)
        rec_flat_gt = torch.matmul(k_vector, self.V_k) + self.s_mean
        rec_gt = rec_flat_gt.view(-1,60,2)
    
        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] >= 2
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        valid_mask_eval = reg_mask[eval_mask]

        gt_eval = gt[eval_mask]
        
            
        
        # pred_modes [num_agents, num_samples, 128]
        self.joint_diffusion.eval()
        num_samples = self.num_eval_samples
        
        goal_point = gt_eval[:,-1,:2].detach().clone()
        
        task = self.guid_task
        if task == 'none':
            cond_gen = None
            grad_guid = None
            
            vel = (gt_eval[:,1:,:2] - gt_eval[:,:-1,:2]).detach().clone()
            vel = torch.abs(vel)
            max_vel = vel.max(-2)[0]
            
            vel = (gt_eval[:,1:,:2] - gt_eval[:,:-1,:2]).detach().clone()
            mean_vel = vel.mean(-2)
            
        elif task == 'goal':
            goal_point = gt_eval[:,-1,:2].detach().clone()
            grad_guid = [goal_point, self.s_mean, self.V_k, self.VT_k, self.latent_mean, self.latent_std]
            cond_gen = None
            
        elif task == 'goal_5s':
            goal_point = gt_eval[:,50,:2].detach().clone()
            grad_guid = [goal_point, self.s_mean, self.V_k, self.VT_k, self.latent_mean, self.latent_std]
            cond_gen = None
            
            
        elif task == 'goal_at5s':
            goal_point = gt_eval[:,-1,:2].detach().clone()
            grad_guid = [goal_point, self.s_mean, self.V_k, self.VT_k, self.latent_mean, self.latent_std]
            cond_gen = None
            
        else:
            raise print('unseen tasks.')
            
        
        guid_method = self.guid_method # none ECM ECMR
        guid_inner_loop = 0 # 111 testing
        guid_param = {}
        guid_param['task'] = task
        guid_param['guid_method'] = guid_method
        cost_param = {'cost_param_costl':self.cost_param_costl, 'cost_param_threl':self.cost_param_threl}
        guid_param['cost_param'] = cost_param
        
        sub_folder = 'important_scenes_with_guid_rs2023' 
        os.makedirs('visual/'+sub_folder, exist_ok=True)
        
        pred_modes = self.joint_diffusion.sample(num_samples, data = data, scene_enc = scene_enc, 
                                                 sampling=self.sampling,
                                                 stride=self.sampling_stride,eval_mask=eval_mask, 
                                                 grad_guid = grad_guid,cond_gen = cond_gen,
                                                 guid_param = guid_param)
        
        if True in torch.isnan(pred_modes):
            print('nan')
            print(data_batch)
            exit()
        
        
        batch_idx = data['agent']['batch'][eval_mask]
        num_scenes = batch_idx[-1].item()+1
        num_agents_per_scene = batch_idx.new_tensor([(batch_idx == i).sum() for i in range(num_scenes)])
        
        
        
        
        pred_modes = self.unnormalize(pred_modes,self.latent_mean, self.latent_std)
        rec_traj = torch.matmul(pred_modes.permute(1,0,2), (self.V_k).unsqueeze(0).repeat(self.num_eval_samples,1,1)) + self.s_mean.unsqueeze(0)
        rec_traj = rec_traj.permute(1,0,2)
        rec_traj = rec_traj.view(rec_traj.size(0), rec_traj.size(1),self.num_generation_timestep,2)
        
        
        plot = (self.guid_plot == 'plot')
        if plot:
            origin_eval = data['agent']['position'][eval_mask, self.init_timestep]
            theta_eval = data['agent']['heading'][eval_mask, self.init_timestep]
            rot_mat = self.create_rot_mat(theta_eval, eval_mask.sum())
            rec_traj_world = torch.matmul(rec_traj[:, :, :, :2],
                                    rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
            
            
            gt_eval_world = torch.matmul(gt_eval[:, :, :2],
                                    rot_mat) + origin_eval[:, :2].reshape(-1, 1, 2)
            gt_eval_world = gt_eval_world.detach().cpu().numpy()
            
            goal_point_world = torch.matmul(goal_point[:, None, None, :],
                                            rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
            goal_point_world = goal_point_world.squeeze(1).squeeze(1)
            goal_point_world = goal_point_world.detach().cpu().numpy()


            img_folder = 'images_g'
            os.makedirs(img_folder,exist_ok=True)
            rec_traj_world = rec_traj_world.detach().cpu().numpy()
            for i in range(num_scenes):
                start_id = torch.sum(num_agents_per_scene[:i])
                end_id = torch.sum(num_agents_per_scene[:i+1])
            
                scenario_id = data['scenario_id'][i]
                base_path_to_data = Path(f'{self.root}/val/raw')
                scenario_folder = base_path_to_data / scenario_id
                
                static_map_path = scenario_folder / f"log_map_archive_{scenario_id}.json"
                scenario_path = scenario_folder / f"scenario_{scenario_id}.parquet"

                scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
                static_map = ArgoverseStaticMap.from_json(static_map_path)
                
                viz_output_dir = Path('visual') / sub_folder
                viz_save_path = viz_output_dir / (task + '_b'+ str(data_batch)+'_s'+str(i)+'_'+guid_method+'_'+self.sampling+'.jpg')
                
                additional_traj = {}
                additional_traj['gt'] = gt_eval_world[start_id:end_id]
                additional_traj['goal_point'] = goal_point_world[start_id:end_id]
                
                additional_traj['rec_traj'] = rec_traj_world[start_id:end_id]
                
                traj_visible = {}
                traj_visible['gt'] = False
                traj_visible['gt_goal'] = False
                traj_visible['goal_point'] = True
                traj_visible['marg_traj'] = False
                traj_visible['rec_traj'] = True
                
                visualize_scenario_prediction(scenario, static_map, additional_traj, traj_visible, viz_save_path)
                
                
    
        

    def test_step(self,
                  data,
                  batch_idx):
        
        
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        
        # pred, scene_enc = self.qcnet(data)
        scene_enc = self.qcnet.encoder.map_encoder(data)
        
        # if self.output_head:
        #     traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
        #                              pred['loc_refine_head'],
        #                              pred['scale_refine_pos'][..., :self.output_dim],
        #                              pred['conc_refine_head']], dim=-1)
        # else:
        #     traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
        #                              pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        # device = traj_refine.device
        # pi = pred['pi']
        device = scene_enc['x_pl'].device
        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] >= 2
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        
        origin_eval = data['agent']['position'][eval_mask, self.init_timestep]
        theta_eval = data['agent']['heading'][eval_mask, self.init_timestep]
        
        rot_mat = self.create_rot_mat(theta_eval, eval_mask.sum())
        # cos, sin = theta_eval.cos(), theta_eval.sin()
        # rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
        # rot_mat[:, 0, 0] = cos
        # rot_mat[:, 0, 1] = sin
        # rot_mat[:, 1, 0] = -sin
        # rot_mat[:, 1, 1] = cos
        # traj_eval = torch.matmul(traj_refine[eval_mask, :, :, :2],
        #                          rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
        
        if self.s_mean == None:
            self.load_vars(device)
            
        # marginal_trajs = traj_refine[eval_mask,:,:,:2]
        # marginal_trajs = marginal_trajs.view(marginal_trajs.size(0),self.num_modes,-1)
        # marginal_mode = torch.matmul((marginal_trajs-self.s_mean.unsqueeze(1)).permute(1,0,2), self.VT_k.unsqueeze(0).repeat(self.num_modes,1,1))
        # marginal_mode = marginal_mode.permute(1,0,2)
        # marginal_mode = self.normalize(marginal_mode, self.latent_mean, self.latent_std)

        
        # mean = marginal_mode.mean(dim=1)
        # std = marginal_mode.std(dim=1) + self.std_reg

        self.joint_diffusion.eval()
        num_samples = self.num_eval_samples

        reverse_steps = 70 # 70
        # no need for vd, but for others it need. Not removing it for now

        pred_modes = self.joint_diffusion.sample(num_samples, data = data, scene_enc = scene_enc, 
                                               sampling=self.sampling,
                                                stride=self.sampling_stride,
                                                reverse_steps=reverse_steps,
                                                eval_mask=eval_mask)
        # blocking below as there is no mode to compare(we don't have marginal_mode)
        # mode_diff = pred_modes.unsqueeze(-2).repeat(1,1,self.num_modes,1) - marginal_mode.unsqueeze(1)
        # [num_agents, num_samples, num_modes]
        # mode_diff = mode_diff.norm(dim=-1)
        
        # # mode_joint_best [num_agents, num_samples]
        # mode_joint_best = torch.argmin(mode_diff,dim=-1)
        
        # joint mode clustering
        # device = mean.device
        # batch_idx = data['agent']['batch'][eval_mask]
        # num_scenes = batch_idx[-1].item()+1
        # # num_agents_per_scene = mode_joint_best.new_tensor([(batch_idx == i).sum() for i in range(num_scenes)])
        # top_modes = torch.randint(0,self.num_modes,(pred_modes.size(0),self.num_modes)).to(pred_modes.device)
        # refine_pi = torch.zeros(top_modes.size(0),self.num_modes).to(pred_modes.device)
        # ### [num_agents, num_samples, 2]
        # joint_goal_pts = traj_refine[eval_mask,:,-1,:2]
        
        # for i in range(num_scenes):
        #     start_id = torch.sum(num_agents_per_scene[:i])
        #     end_id = torch.sum(num_agents_per_scene[:i+1])
            
        #     # initialize the modes for single agent
        #     if end_id - start_id == 1:
        #         for j in range(self.num_modes):
        #             top_modes[start_id:end_id,j] = j
            
        #     # cluster the modes
        #     topk_keys = []
        #     topk_nums = []
        #     topk_joint_modes = []
        #     for j in range(num_samples):
        #         key = 'k'
        #         for it in mode_joint_best[start_id:end_id,j]:
        #             key += str(it.cpu().numpy())
                
        #         try:
        #             idx = topk_keys.index(key)
        #             topk_nums[idx] += 1
        #         except ValueError:
        #             topk_keys.append(key)
        #             topk_nums.append(1)
        #             topk_joint_modes.append(mode_joint_best[start_id:end_id,j:j+1])

        #     topk_nums = torch.tensor(topk_nums).to(device)
        #     topk_joint_modes = torch.cat(topk_joint_modes, dim=1)
            
        #     # sort
        #     ids = torch.argsort(topk_nums, descending=True)
            
        #     if self.cluster == 'normal':
        #         topk_ids = ids[:self.num_modes]
        #         total_num = torch.sum(topk_nums[topk_ids])
        #         top_modes[start_id:end_id,:topk_ids.size(0)] = topk_joint_modes[:, topk_ids]
        #         refine_pi[start_id:end_id,:topk_ids.size(0)] = topk_nums[topk_ids]/total_num
                
        #     elif self.cluster == 'traj':
        #         ids_init = ids
        #         nums_init = topk_nums
        #         # print('init',nums_init[ids_init])
                
        #         # based on sort, cluster in the trajectory space
        #         ### [num_agents, num_samples, 2]
        #         scene_joint_goal_pts = joint_goal_pts[start_id:end_id]
        #         num_agents = scene_joint_goal_pts.size(0)
        #         max_threshold = self.cluster_max_thre
        #         # print(max_threshold)
        #         mean_threshold = self.cluster_mean_thre
        #         queue = torch.ones(ids.shape[0],dtype=torch.int8)
        #         nms_nums = []
        #         nms_idx = []
        #         for j in range(ids.size(0)):
        #             if queue[j] == 0:
        #                 continue
        #             queue[j] = 0
        #             idx = ids[j]
        #             nms_idx.append(idx)
        #             temp_nums = topk_nums[idx]
        #             temp_modes = topk_joint_modes[:,idx]
        #             target_jgp = scene_joint_goal_pts[torch.arange(num_agents), temp_modes]
                                        
        #             # queue_ids = torch.arange(j+1,ids.size(0)).to(device)
        #             queue_ids = torch.nonzero(queue).squeeze(1).to(device)
                    
                    
        #             cand_ids = ids[queue_ids]        
        #             num_cands = cand_ids.size(0)
        #             cand_modes = topk_joint_modes[:,cand_ids]
        #             cand_jgp = scene_joint_goal_pts[torch.arange(num_agents).unsqueeze(-1).repeat(1,num_cands), cand_modes]
        #             diff =  torch.norm(target_jgp.unsqueeze(1) - cand_jgp,dim=-1)
        #             max_diff = torch.max(diff, dim=0)[0]
        #             group_ids = torch.nonzero(max_diff < max_threshold).squeeze(1)
        #             cand_nums = topk_nums[cand_ids[group_ids]]
        #             queue[queue_ids[group_ids]] = 0
        #             nms_nums.append(temp_nums + torch.sum(cand_nums))
                    
                    
        #         # when clustering into less than num_modes groups, add more groups
        #         if len(nms_idx) < self.num_modes:
        #             for idx in ids_init:
        #                 if idx not in nms_idx:
        #                     nms_idx.append(idx)
        #                     nms_nums.append(nums_init[idx])
        #                     if len(nms_idx) == self.num_modes:
        #                         break
                    
        #         # assign modes
        #         nms_idx = torch.tensor(nms_idx).to(device)
        #         nms_nums = torch.tensor(nms_nums).to(device)
        #         ids = torch.argsort(nms_nums, descending=True)
        #         topk_ids = ids[:self.num_modes]
                
        #         lack_nms = self.num_modes - topk_ids.size(0)
        #         if end_id - start_id == 1 and lack_nms > 0:
        #             total_num = torch.sum(nms_nums[topk_ids]) + lack_nms
        #             top_modes[start_id:end_id,:topk_ids.size(0)] = topk_joint_modes[:, nms_idx[topk_ids]]
        #             refine_pi[start_id:end_id,:topk_ids.size(0)] = nms_nums[topk_ids]/total_num
        #             for k in range(lack_nms):
        #                 for j in range(self.num_modes):
        #                     if j not in top_modes[start_id:end_id,:topk_ids.size(0)+k]:
        #                         top_modes[start_id:end_id,topk_ids.size(0)+k]=j
        #                         refine_pi[start_id:end_id,topk_ids.size(0)+k] = 1/total_num
                    
                    
        #         else:
        #             total_num = torch.sum(nms_nums[topk_ids])
        #             top_modes[start_id:end_id,:topk_ids.size(0)] = topk_joint_modes[:, nms_idx[topk_ids]]
        #             refine_pi[start_id:end_id,:topk_ids.size(0)] = nms_nums[topk_ids]/total_num
                    
        
        # num_mm = top_modes.size(1)
        # traj_refine_topk = traj_eval[torch.arange(traj_eval.size(0)).unsqueeze(-1).repeat(1,num_mm), top_modes][...,:self.output_dim]
        # traj_eval = traj_refine_topk.cpu().numpy()
        # pi_eval = refine_pi.cpu().numpy()
        
        
        # if self.dataset == 'argoverse_v2':
        #     eval_id = list(compress(list(chain(*data['agent']['id'])), eval_mask))
        #     if isinstance(data, Batch):
        #         for i in range(data.num_graphs):
        #             start_id = torch.sum(num_agents_per_scene[:i])
        #             end_id = torch.sum(num_agents_per_scene[:i+1])
        #             scenario_trajectories = {}
        #             for j in range(start_id,end_id):
        #                 scenario_trajectories[eval_id[j]] = traj_eval[j]
        #             self.test_predictions[data['scenario_id'][i]] = (pi_eval[start_id,:], scenario_trajectories)
        #     else:
        #         self.test_predictions[data['scenario_id']] = (pi_eval[0], {eval_id[0]: traj_eval[0]})
        # else:
        #     raise ValueError('{} is not a valid dataset'.format(self.dataset))


    def on_test_end(self):
        if self.dataset == 'argoverse_v2':
            ChallengeSubmission(self.test_predictions).to_parquet(
                Path(self.submission_dir) / f'{self.submission_file_name}.parquet')
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        # assert len(inter_params) == 0
        # assert len(param_dict.keys() - union_params) == 0
        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]
        # optim_groups_diff = [
        #     {"params": [param_dict[param_name] for param_name in sorted(list(decay)) if 'map_encoder' not in param_name],
        #      "weight_decay": self.weight_decay},
        #     {"params": [param_dict[param_name] for param_name in sorted(list(no_decay)) if 'map_encoder' not in param_name],
        #      "weight_decay": 0.0},
        # ]

        # optim_groups_qcnet_map = [
        #     {"params": [param_dict[param_name] for param_name in sorted(list(decay)) if 'map_encoder' in param_name],
        #      "weight_decay": self.weight_decay},
        #     {"params": [param_dict[param_name] for param_name in sorted(list(no_decay)) if 'map_encoder' in param_name],
        #      "weight_decay": 0.0},
        # ]

        # optimizer_diff = torch.optim.AdamW(optim_groups_diff, lr=self.lr, weight_decay=self.weight_decay)
        # scheduler_diff = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_diff, T_max=self.T_max, eta_min=0.0)

        optimizer = torch.optim.AdamW(optim_groups, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=self.lr, steps_per_epoch=self.trainer.estimated_stepping_batches // self.trainer.max_epochs,  # Or len(train_dataloader) if you know it
            epochs=self.trainer.max_epochs)

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=100, T_mult= 2, eta_min=1e-6)
        # scheduler = CosineAnnealingWarmupRestarts(
        #     optimizer,
        #     first_cycle_steps=100,
        #     cycle_mult=2.0,
        #     max_lr=self.lr,
        #     min_lr=1e-7,
        #     warmup_steps=10,
        #     gamma=0.9,
        # )

        
        return [optimizer], [{
        'scheduler': scheduler,
        'interval': 'step',  # or 'epoch', depending on when you want to step the scheduler
        'frequency': 1
        }]
    
    # def set_opt_lr(self, lr):
    #     [optimizer], [scheduler] = self.optimizers()
    #     for g in optimizer.param_groups:
    #         g['lr'] = 0.001
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('QCNet')
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--input_dim', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--output_dim', type=int, default=2)
        parser.add_argument('--output_head', action='store_true')
        parser.add_argument('--init_timestep', type=int, default=50)
        parser.add_argument('--num_generation_timestep', type=int, default=60)
        parser.add_argument('--num_freq_bands', type=int, default=64)
        parser.add_argument('--num_map_layers', type=int, default=1)
        parser.add_argument('--num_agent_layers', type=int, default=2)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--head_dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--pl2pl_radius', type=float, required=True)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--submission_dir', type=str, default='./')
        parser.add_argument('--submission_file_name', type=str, default='submission')
        parser.add_argument('--qcnet_map_ckpt_path', type=str, required=True)
        parser.add_argument('--num_denoiser_layers', type=int, default=3)
        parser.add_argument('--num_diffusion_steps', type=int, default=10)
        parser.add_argument('--beta_1', type=float, default=1e-4)
        parser.add_argument('--beta_T', type=float, default=0.05)
        parser.add_argument('--diff_type', choices=['opsd', 'opd', 'vd']) 
        parser.add_argument('--sampling', choices=['ddpm','ddim'])
        parser.add_argument('--sampling_stride', type = int, default = 20)
        parser.add_argument('--num_eval_samples', type = int, default = 6)
        parser.add_argument('--train_agent', choices=['all', 'eval'],default = 'all')
        parser.add_argument('--path_pca_s_mean', type = str,default = 'pca/s_mean_10.npy')
        parser.add_argument('--path_pca_VT_k', type = str,default = 'pca/VT_k_10.npy')
        parser.add_argument('--path_pca_latent_mean', type = str,default = 'pca/latent_mean_10.npy')
        parser.add_argument('--path_pca_latent_std', type = str,default = 'pca/latent_std_10.npy')
        parser.add_argument('--m_dim', type = int,default = 10)
        
        return parent_parser
