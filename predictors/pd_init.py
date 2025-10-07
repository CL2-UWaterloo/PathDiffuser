# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
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
import numpy as np

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from metrics import  OffRoad, Collision, NearestEdge
from metrics import JSD_SPEED, JSD_MAP_DIST, JSD_MAP_ANGLE, JSD_INTERACTIVE, JSD_MAP_INTERSECTION, JSD_MAP_INTERSECTION_ANGLE, JSD_LOCAL_DENSITY

from modules import InitDiffusion
from modules import QCNetMapEncoderPT as QCNetMapEncoder


from av2.datasets.motion_forecasting import scenario_serialization
from visualization import *
from av2.map.map_api import ArgoverseStaticMap

import os

        
class PDInit(pl.LightningModule):

    def __init__(self,
                 args,
                 **kwargs) -> None:
        super(PDInit, self).__init__()
        
        self.save_hyperparameters()
        self.dataset = args.dataset
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.output_head = args.output_head
        self.init_timestep = args.init_timestep
        # self.num_generation_timestep = args.num_generation_timestep
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
                                                init_timestep=0,
                                                pl2pl_radius=self.pl2pl_radius,
                                                num_freq_bands=self.num_freq_bands,
                                                num_layers=self.num_map_layers,
                                                num_heads=self.num_heads,
                                                head_dim=self.head_dim,
                                                dropout=self.dropout)

        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.linear = nn.Linear(10,2)
        
        self.joint_diffusion = InitDiffusion(args=args)

        self.OffRoad = OffRoad()
        self.Collision = Collision()
        self.NearestEdge = NearestEdge()


        self.OffRoad_gt = OffRoad()
        self.Collision_gt = Collision()
        self.NearestEdge_gt = NearestEdge()

        self.JSD_MAP_DIST = JSD_MAP_DIST()
        self.JSD_MAP_ANGLE = JSD_MAP_ANGLE()
        self.JSD_LOCAL_DENSITY = JSD_LOCAL_DENSITY()
        self.JSD_INTERACTIVE = JSD_INTERACTIVE()
        self.JSD_SPEED = JSD_SPEED()
        self.JSD_MAP_INTERSECTION = JSD_MAP_INTERSECTION()
        self.JSD_MAP_INTERSECTION_ANGLE = JSD_MAP_INTERSECTION_ANGLE()

        self.num_all_agents = 0
        self.M_dis = []
        self.order_ac = []
        
        self.eval_line = torch.tensor([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=torch.float32)
        self.dist_from_gt_all = torch.tensor([0], dtype=torch.float32)
        self.cnt = 0
        
    def add_extra_param(self, args):
        self.guid_sampling = args.guid_sampling
        self.joint_diffusion.guid_sampling = args.guid_sampling
        self.guid_task = args.guid_task
        self.guid_method = args.guid_method
        self.guid_plot = args.guid_plot
        self.plot = args.plot
        self.path_pca_V_k = args.path_pca_V_k
        self.V_k = None

        self.cond_norm = args.cond_norm
        self.cost_param_costl = args.cost_param_costl
        self.cost_param_threl = args.cost_param_threl
        self.root = args.root
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
        scene_enc = self.qcnet_mapencoder(data)
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
        # interpolation_mask[:, -1] = False
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
    def create_rot_mat_cossin(self, cos_sin, num_agents):
        cos, sin = cos_sin[..., 0].squeeze(-1), cos_sin[..., 1].squeeze(-1)
        mat = torch.zeros(num_agents, 2, 2, device=cos_sin.device)
        mat[:, 0, 0] = cos
        mat[:, 0, 1] = -sin
        mat[:, 1, 0] = sin
        mat[:, 1, 1] = cos
        return mat
    

    def training_step(self,
                      data,
                      batch_idx):
        
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        
        reg_mask = data['agent']['predict_mask'][:, self.init_timestep:]
        
        scene_enc = self.qcnet_mapencoder(data)

        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)

        eval_mask = data['agent']['mask']
        
        gt = gt[eval_mask][..., :self.output_dim]

        reg_mask = reg_mask[eval_mask]
        
        num_samples = 1
        
        init_trans = data['agent']['init_translation'][eval_mask][:,:2]
        
        delta_rot = (data['agent']['init_angle'][eval_mask]).unsqueeze(-1)
        
        head_cosine = torch.cat([delta_rot.cos(), delta_rot.sin()], dim=-1)
        init_speed = data['agent']['init_speed'][eval_mask].unsqueeze(-1)
        m_init = torch.cat([init_trans, head_cosine, init_speed], dim=-1)

        loss_diff_init, pred_init = self.joint_diffusion.get_loss(m_init, data = data, scene_enc = scene_enc,eval_mask=eval_mask, num_samples=num_samples)

        pred_trans, pred_head, pred_speed = pred_init[..., :2], pred_init[..., 2:4], pred_init[..., 4]

        
        target_origin = data['agent']['init_translation'][eval_mask].unsqueeze(1).repeat(1, num_samples, 1)[..., :2]
        target_theta = data['agent']['init_angle'][eval_mask].unsqueeze(1).repeat(1, num_samples)
        agent_batch = data['agent']['batch'][eval_mask]
        map_min = data['map_min'].view(-1, 3)[..., :2].unsqueeze(1)[agent_batch]
        map_max = data['map_max'].view(-1, 3)[..., :2].unsqueeze(1)[agent_batch]

        pred_trans = (pred_trans + 1) * (map_max - map_min) / 2 + map_min
        target_origin = (target_origin + 1) * (map_max - map_min) / 2 + map_min

        target_theta = torch.cat([target_theta.cos(), target_theta.sin()], dim=1)
        
        loss_trans = torch.nn.HuberLoss()(pred_trans, target_origin)
        loss_rot2 = torch.nn.HuberLoss()(pred_head.view(target_theta.shape), target_theta)    
        loss_speed = torch.nn.HuberLoss()(pred_speed, data['agent']['init_speed'][eval_mask].unsqueeze(1).repeat(1, num_samples))

        self.log('train_lr', self.optimizers().param_groups[0]['lr'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        
    
        loss_diff_trans= loss_diff_init[...,:2].mean()
        loss_diff_theta = loss_diff_init[...,2:4].mean()
        loss_diff_speed = loss_diff_init[...,4].mean()
        loss_diff_init = loss_diff_init.mean()

        self.log('train/loss_diff_init', loss_diff_init, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        
        self.log('train/loss_diff_trans', loss_diff_trans, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train/loss_diff_theta', loss_diff_theta, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train/loss_diff_speed', loss_diff_speed, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        
        self.log('train/trans_loss', loss_trans, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train/rot_loss2', loss_rot2, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train/speed_loss', loss_speed, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

        return loss_diff_init


    def validation_step(self,
                    data,
                    batch_idx):
        if self.guid_sampling == 'no_guid':
            self.validation_step_norm(data, batch_idx)
        elif self.guid_sampling == 'guid':
            self.validation_step_guid(data, batch_idx)
        
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
        
        scene_enc = self.qcnet_mapencoder(data)
    
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)

        if self.s_mean == None:
            self.load_vars(self.device)
        mask = data['agent']['mask']
        
        gt_n = gt[mask][..., :self.output_dim]
        device = gt_n.device
        

        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['mask']
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))


        gt_eval = gt[eval_mask]
    
 
        self.joint_diffusion.eval()
        num_samples = self.num_eval_samples

        if_output_diffusion_process = False
        

        if if_output_diffusion_process:
            reverse_steps = self.num_diffusion_steps
            pred_modes, pred_delta = self.joint_diffusion.sample(num_samples, data = data, scene_enc = scene_enc, 
                                                    sampling=self.sampling,
                                                    stride=self.sampling_stride,eval_mask=eval_mask,
                                                    if_output_diffusion_process=if_output_diffusion_process,
                                                    reverse_steps=reverse_steps)

            
            inter_latents = pred_modes[::1]
            inter_trajs = []
            for latent in inter_latents:
                inter_trajs.append(latent)
            
            
            
            pred_modes = pred_modes[-1]

        else:
            
            reverse_steps = None
            pred_init = self.joint_diffusion.sample(num_samples, data = data, scene_enc = scene_enc, 
                                                    sampling=self.sampling,
                                                    stride=self.sampling_stride,eval_mask=eval_mask,
                                                    if_output_diffusion_process=if_output_diffusion_process,
                                                    reverse_steps=reverse_steps)
        agent_batch = data['agent']['batch'][eval_mask]

        pred_trans, pred_head, pred_speed = pred_init[..., :2], pred_init[..., 2:4], pred_init[..., 4]
        
   
        map_min = data['map_min'].view(-1, 3)[..., :2].unsqueeze(1)[agent_batch]
        map_max = data['map_max'].view(-1, 3)[..., :2].unsqueeze(1)[agent_batch]

        pred_trans = (pred_trans + 1) * (map_max - map_min) / 2 + map_min
        
        if True in torch.isnan(pred_trans):
            print('nan')
            print(data_batch)
            exit()

        # joint mode clustering
        batch_idx = data['agent']['batch'][eval_mask]
        num_scenes = batch_idx[-1].item()+1
        num_agents_per_scene = pred_trans.new_tensor([(batch_idx == i).sum() for i in range(num_scenes)]).type(torch.int64)


        all_rel_origin_eval = data['agent']['position'][eval_mask, self.init_timestep, :2]
        all_rel_theta_eval = data['agent']['heading'][eval_mask, self.init_timestep]
        all_rel_rot_mat = self.create_rot_mat(all_rel_theta_eval, all_rel_theta_eval.shape[0])
        all_rel_rot_mat_inv = all_rel_rot_mat.permute(0, 2, 1)
        
        pred_rot_mat = self.create_rot_mat_cossin(pred_head, pred_head.shape[0])
        pred_rot_mat_inv = pred_rot_mat.permute(0, 2, 1)
        

        if self.eval_line.device != device:
            self.eval_line = self.eval_line.to(device)
        rec_traj_world = torch.matmul(self.eval_line.repeat(gt_eval.shape[0], 1, 1),
                                pred_rot_mat_inv).unsqueeze(1) + pred_trans.reshape(-1, 1, 1, 2)
        gt_eval_world = torch.matmul(self.eval_line.repeat(gt_eval.shape[0], 1, 1),
                                all_rel_rot_mat_inv) + all_rel_origin_eval[:, :2].reshape(-1, 1, 2)
            
        trans_loss = torch.nn.MSELoss()(pred_trans.squeeze(1), all_rel_origin_eval)
        dist_from_gt = torch.norm(pred_trans.squeeze(1) - all_rel_origin_eval, dim=-1)
        self.dist_from_gt_all += dist_from_gt.sum().item()
        self.cnt += dist_from_gt.size(0)
        
        self.log('val_trans_loss', trans_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
        agent_batch = data['agent']['batch'][eval_mask]
        pred_angle = torch.atan2(pred_head[..., 1], pred_head[..., 0])
        gt_init = torch.cat([all_rel_origin_eval, all_rel_theta_eval.unsqueeze(-1)], dim=-1)
        pred_init = torch.cat([pred_trans, pred_angle.unsqueeze(-1)], dim=-1)
        
        self.JSD_LOCAL_DENSITY.update(pred=pred_init, gt=gt_init, agent_batch=agent_batch, data=data)
        self.log('val/jsd_local_density', self.JSD_LOCAL_DENSITY, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
        self.JSD_MAP_DIST.update(pred=pred_init, gt=gt_init, agent_batch=agent_batch, map_pts=data['map_point'])
        self.log('val/jsd_map_dist', self.JSD_MAP_DIST, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
        hist_angle_gt, hist_angle_pred = self.JSD_MAP_DIST.get_angle_hist()

        self.JSD_MAP_ANGLE.update(hist_angle_gt, hist_angle_pred)
        self.log('val/jsd_map_angle', self.JSD_MAP_ANGLE, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
        
        self.JSD_INTERACTIVE.update(pred_init, gt_init, agent_batch=agent_batch)
        self.log('val/jsd_interactive', self.JSD_INTERACTIVE, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)

        gt_init_speed = data['agent']['init_speed'][eval_mask]
        self.JSD_SPEED.update(pred_speed, gt_init_speed)
        self.log('val/jsd_speed', self.JSD_SPEED, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)


        self.OffRoad.update(pred=pred_trans, agent_batch=agent_batch, map_pts=data['map_point'])
        self.log('val/offroad_rate', self.OffRoad, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)

        self.NearestEdge.update(pred=pred_trans, agent_batch=agent_batch, map_pts=data['map_point'])
        self.log('val/nearest_edge_dist', self.NearestEdge, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)

        self.Collision.update(pred=pred_trans, agent_batch=agent_batch, agent_type=data['agent']['type'][eval_mask])
        self.log('val/collision_rate', self.Collision, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)


        self.OffRoad_gt.update(pred=all_rel_origin_eval,agent_batch=agent_batch, map_pts=data['map_point'])
        self.log('val/offroad_rate_gt', self.OffRoad_gt, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)

        self.NearestEdge_gt.update(pred=all_rel_origin_eval,agent_batch=agent_batch, map_pts=data['map_point'])
        self.log('val/nearest_edge_dist_gt', self.NearestEdge_gt, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)

        self.Collision_gt.update(pred=all_rel_origin_eval, agent_batch=agent_batch, agent_type=data['agent']['type'][eval_mask])
        self.log('val/collision_rate_gt', self.Collision_gt, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
        if print_flag:
            print(f'GT: collision: {self.Collision_gt.compute().item()}, nearest_edge:{self.NearestEdge_gt.compute().item()}, offroad:{self.OffRoad_gt.compute().item()}')
            mean_dist = self.dist_from_gt_all/self.cnt
            print(f'Gen: collision: {self.Collision.compute().item()}, nearest_edge:{self.NearestEdge.compute().item()}, offroad:{self.OffRoad.compute().item()}, dist: {mean_dist}')
        
        scenario_id = data['scenario_id'][0] 

        goal_point = gt_eval[:,-1,:2]
        plot = self.plot
        if plot == 'plot':
            if if_output_diffusion_process:
                inter_trajs_world = []
                rot_mat = all_rel_rot_mat_inv
                origin_eval = all_rel_origin_eval
              
                for traj in inter_trajs:
                    traj_world = torch.matmul(traj[:, :, :, :2],
                                    rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
                    inter_trajs_world.append(traj_world.detach().cpu().numpy())

            
            gt_eval_world = gt_eval_world.detach().cpu().numpy()
            
            goal_point_world = torch.matmul(goal_point[:, None, None, :],
                                            all_rel_rot_mat_inv.unsqueeze(1)) + all_rel_origin_eval[:, :2].reshape(-1, 1, 1, 2)
            goal_point_world = goal_point_world.squeeze(1).squeeze(1)
            goal_point_world = goal_point_world.detach().cpu().numpy()


            img_folder = 'visual'
            if self.ckpt_path:
                sub_folder = self.ckpt_path.split('/')[-3]
            else:
                sub_folder = 'tmp'
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

                viz_save_path = viz_output_dir / ('b'+ str(data_batch)+'_s'+str(i)+'_'+self.sampling+'.jpg')
                
                additional_traj = {}
                additional_traj['gt'] = gt_eval_world[start_id:end_id]
                additional_traj['rec_traj'] = rec_traj_world[start_id:end_id]
                
                traj_visible = {}
                                
                traj_visible['gt'] = False
                traj_visible['gt_goal'] = False
                traj_visible['goal_point'] = False
                traj_visible['rec_traj'] = True
                                
                visualize_scenario_prediction(scenario, static_map, additional_traj, traj_visible, viz_save_path, data)



    def validation_step_guid(self,
                        data,
                        batch_idx):
        print_flag = False
        if batch_idx % 1 == 0:
            print_flag = True
        
        data_batch = batch_idx
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        
        scene_enc = self.qcnet_mapencoder(data)
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        
        if self.s_mean == None:
            self.load_vars(gt.device)
            

        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['mask']
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

        gt_eval = gt[eval_mask]
        

        self.joint_diffusion.eval()
        num_samples = self.num_eval_samples
        
        task = self.guid_task
        if task == 'none':
            cond_gen = None
            grad_guid = None
            
            vel = (gt_eval[:,1:,:2] - gt_eval[:,:-1,:2]).detach().clone()
            vel = torch.abs(vel)
            max_vel = vel.max(-2)[0]
            
            vel = (gt_eval[:,1:,:2] - gt_eval[:,:-1,:2]).detach().clone()
            mean_vel = vel.mean(-2)
                    
        elif task == 'map':
            goal_point = gt_eval[:,-1,:2].detach().clone()
            
            grad_guid = [data, self.s_mean, self.V_k, self.VT_k, self.latent_mean, self.latent_std]
            cond_gen = None

        elif task == 'map_collision':
            
            grad_guid = [data, self.s_mean, self.V_k, self.VT_k, self.latent_mean, self.latent_std]
            cond_gen = None
        elif task == 'original':
            
            grad_guid = [data, self.s_mean, self.V_k, self.VT_k, self.latent_mean, self.latent_std]
            cond_gen = None
        else:
            raise print('unseen tasks.')
            
        
        guid_method = self.guid_method # none ECM ECMR
        guid_param = {}
        guid_param['task'] = task
        guid_param['guid_method'] = guid_method
        cost_param = {'cost_param_costl':self.cost_param_costl, 'cost_param_threl':self.cost_param_threl}
        guid_param['cost_param'] = cost_param
        
        sub_folder = self.ckpt_path.split('/')[-3] 
        os.makedirs('visual/'+sub_folder, exist_ok=True)
        
        pred_init = self.joint_diffusion.sample(num_samples, data = data, scene_enc = scene_enc, 
                                                 sampling=self.sampling,
                                                 stride=self.sampling_stride,eval_mask=eval_mask, 
                                                 grad_guid = grad_guid,cond_gen = cond_gen,
                                                 guid_param = guid_param)
        
        if True in torch.isnan(pred_init):
            print('nan')
            print(data_batch)
            exit()
        
        pred_trans, pred_head, pred_speed = pred_init[..., :2], pred_init[..., 2:4], pred_init[..., 4]

        batch_idx = data['agent']['batch'][eval_mask]
        map_min = data['map_min'].view(-1, 3)[..., :2].unsqueeze(1)[batch_idx]
        map_max = data['map_max'].view(-1, 3)[..., :2].unsqueeze(1)[batch_idx]
        pred_trans = (pred_trans + 1) * (map_max - map_min) / 2 + map_min
        device = pred_trans.device
        num_scenes = batch_idx[-1].item()+1
        num_agents_per_scene = batch_idx.new_tensor([(batch_idx == i).sum() for i in range(num_scenes)])

        all_rel_origin_eval = data['agent']['position'][eval_mask, self.init_timestep, :2]
        all_rel_theta_eval = data['agent']['heading'][eval_mask, self.init_timestep]
        all_rel_rot_mat = self.create_rot_mat(all_rel_theta_eval, all_rel_theta_eval.shape[0])
        all_rel_rot_mat_inv = all_rel_rot_mat.permute(0, 2, 1)
        
        pred_rot_mat = self.create_rot_mat_cossin(pred_head, pred_head.shape[0])
        pred_rot_mat_inv = pred_rot_mat.permute(0, 2, 1)
        
        if self.eval_line.device != device:
            self.eval_line = self.eval_line.to(device)
        rec_traj_world = torch.matmul(self.eval_line.repeat(gt_eval.shape[0], 1, 1),
                                pred_rot_mat_inv).unsqueeze(1) + pred_trans.reshape(-1, 1, 1, 2)
        gt_eval_world = torch.matmul(self.eval_line.repeat(gt_eval.shape[0], 1, 1),
                                all_rel_rot_mat_inv) + all_rel_origin_eval[:, :2].reshape(-1, 1, 2)
        trans_loss = torch.nn.MSELoss()(pred_trans.squeeze(1), all_rel_origin_eval)
        dist_from_gt = torch.norm(pred_trans.squeeze(1) - all_rel_origin_eval, dim=-1)
        self.dist_from_gt_all += dist_from_gt.sum().item()
        self.cnt += dist_from_gt.size(0)

        self.log('val_trans_loss', trans_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
        agent_batch = data['agent']['batch'][eval_mask]

        self.OffRoad.update(pred=pred_trans, agent_batch=agent_batch,scenario_id_list=data['scenario_id'])
        self.log('val/offroad_rate', self.OffRoad, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)

        self.NearestEdge.update(pred=pred_trans, agent_batch=agent_batch,scenario_id_list=data['scenario_id'])
        self.log('val/nearest_edge_dist', self.NearestEdge, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)

        self.Collision.update(pred=pred_trans, agent_batch=agent_batch, agent_type=data['agent']['type'][eval_mask])
        self.log('val/collision_rate', self.Collision, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)


        self.OffRoad_gt.update(pred=all_rel_origin_eval,agent_batch=agent_batch, scenario_id_list=data['scenario_id'])
        self.log('val/offroad_rate_gt', self.OffRoad_gt, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)

        self.NearestEdge_gt.update(pred=all_rel_origin_eval,agent_batch=agent_batch, scenario_id_list=data['scenario_id'])
        self.log('val/nearest_edge_dist_gt', self.NearestEdge_gt, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)

        self.Collision_gt.update(pred=all_rel_origin_eval, agent_batch=agent_batch, agent_type=data['agent']['type'][eval_mask])
        self.log('val/collision_rate_gt', self.Collision_gt, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
        if print_flag:
            print(f'GT: collision: {self.Collision_gt.compute().item()}, nearest_edge:{self.NearestEdge_gt.compute().item()}, offroad:{self.OffRoad_gt.compute().item()}')
            mean_dist = self.dist_from_gt_all/self.cnt
            print(f'Gen: collision: {self.Collision.compute().item()}, nearest_edge:{self.NearestEdge.compute().item()}, offroad:{self.OffRoad.compute().item()}, dist: {mean_dist}')
            
        
        plot = (self.guid_plot == 'plot')
        if plot:
            
            gt_eval_world = gt_eval_world.detach().cpu().numpy()
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
                additional_traj['rec_traj'] = rec_traj_world[start_id:end_id]
                
                traj_visible = {}
                traj_visible['gt'] = True
                traj_visible['gt_goal'] = False
                traj_visible['goal_point'] = False
                traj_visible['marg_traj'] = False
                traj_visible['rec_traj'] = True
                
                visualize_scenario_prediction(scenario, static_map, additional_traj, traj_visible, viz_save_path, data)
                

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
        
        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, weight_decay=self.weight_decay, lr=self.lr)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=self.lr, steps_per_epoch=self.trainer.estimated_stepping_batches // self.trainer.max_epochs,  # Or len(train_dataloader) if you know it
            epochs=self.trainer.max_epochs)
                

        return [optimizer], [{
        'scheduler': scheduler,
        'interval': 'step',  # or 'epoch', depending on when you want to step the scheduler
        'frequency': 1
        }]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('QCNet')
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--input_dim', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--output_dim', type=int, default=2)
        parser.add_argument('--output_head', action='store_true')
        parser.add_argument('--init_timestep', type=int, default=50)
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
        parser.add_argument('--qcnet_map_ckpt_path', type=str, required=False)
        parser.add_argument('--num_denoiser_layers', type=int, default=3)
        parser.add_argument('--num_diffusion_steps', type=int, default=10)
        parser.add_argument('--beta_1', type=float, default=1e-4)
        parser.add_argument('--beta_T', type=float, default=0.05)
        parser.add_argument('--diff_type', choices=['opsd', 'opd', 'vd']) 
        parser.add_argument('--sampling', choices=['ddpm','ddim'])
        parser.add_argument('--sampling_stride', type = int, default = 20)
        parser.add_argument('--num_eval_samples', type = int, default = 6)
        parser.add_argument('--train_agent', choices=['all', 'eval'],default = 'all')
        
        return parent_parser
