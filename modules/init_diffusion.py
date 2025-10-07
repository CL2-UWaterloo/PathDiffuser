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
import math
import copy
import numpy as np

from typing import Dict, Mapping, Optional
from pynvml import *
nvmlInit()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Bernoulli

from layers import TransformerDecoderLayerDiff
from layers.sinusoidal_embedding import sinusoidal_embedding

from layers import FourierEmbedding
from utils import weight_init

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


class InitDiffusion(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.diff_type = args.diff_type
        self.guid_sampling = args.guid_sampling
        
        self.net = InitDenoiser(
            dataset=args.dataset,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            output_head=args.output_head,
            init_timestep=args.init_timestep,
            num_freq_bands=args.num_freq_bands,
            num_layers=args.num_denoiser_layers,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            dropout=args.dropout,
            diff_type=args.diff_type,
            m_dim = args.m_dim
        )
        
        self.var_sched = VarianceSchedule(
                num_steps = args.num_diffusion_steps,
                beta_1 = args.beta_1,
                beta_T = args.beta_T,
                mode = 'linear'
            )
        self.infer_time_per_step = []
        self.GPU_incre_memory = []
        # self.t_std = []
        # self.t_mean = []

        probs = torch.tensor([0.5])
        self.B_dist = Bernoulli(probs=probs)



    def get_loss(self, 
                 diff_input,
                 data: HeteroData,
                 scene_enc: Mapping[str, torch.Tensor],
                 eval_mask = None,
                 num_samples = None) -> Dict[str, torch.Tensor]:
        
        return self.get_loss_vd(diff_input,data,scene_enc,eval_mask, num_samples,)
        
    def get_loss_vd(self, 
                 m_init,
                 data: HeteroData,
                 scene_enc: Mapping[str, torch.Tensor],
                 eval_mask = None,
                 num_samples = 1,) -> Dict[str, torch.Tensor]:
        # m: [num_agents, d_latent]
        
        x_init_0 = m_init.unsqueeze(1).repeat(1,num_samples,1)

        device = m_init.device
        batch_idx = data['agent']['batch'][eval_mask]
        num_agents = eval_mask.sum()# mean.size(0)
        num_dim = self.net.m_dim
        num_scenes = batch_idx[-1].item()+1

        agent_batch = data['agent']['batch'][eval_mask]
        t = torch.tensor(self.var_sched.uniform_sample_t(num_scenes)).to(device)
        
        alpha_bar = self.var_sched.alpha_bars[t][:,None].to(device)
        beta = self.var_sched.betas[t][:,None].to(device)[agent_batch]
        c0 = torch.sqrt(alpha_bar).unsqueeze(-1).repeat(1,num_samples, 1)
        c1 = torch.sqrt(1 - alpha_bar).unsqueeze(-1).repeat(1,num_samples, 1)
        
        e_init_rand = torch.randn_like(x_init_0).to(device) 

        x_init_t = c0[agent_batch] * x_init_0 + c1[agent_batch] * e_init_rand
        mode = self.B_dist.sample()
        # now delta_rot_pred is angle! add the ego initial angle, then we can get the heading relative to its own
        g_init_theta = self.net(copy.deepcopy(x_init_t), beta, data, scene_enc, num_samples = num_samples, eval_mask=eval_mask, mode=mode)
        
        loss_init = ((e_init_rand- g_init_theta) ** 2)#.mean()

        x_init_0_reconstructed = (x_init_t - c1[agent_batch]  * g_init_theta) / c0[agent_batch]
        return loss_init, x_init_0_reconstructed
    
    
    def sample(self, 
               num_samples: int,
               data: HeteroData,
               scene_enc: Mapping[str, torch.Tensor],
               if_output_diffusion_process = False,
               start_data = None,
               reverse_steps = None,
               eval_mask = None,
               sampling="ddpm", 
               stride=20,
               grad_guid = None,
               cond_gen = None,
               guid_param = None,
               uc = None,
               clean_data = None,
               ) -> Dict[str, torch.Tensor]:
        if self.guid_sampling == 'guid':
            
            return self.sample_guide(num_samples, data, scene_enc, 
                                    if_output_diffusion_process, start_data,reverse_steps,
                                    eval_mask, sampling, stride, grad_guid, guid_param=guid_param)
        else:
            return self.sample_vd(num_samples, data, scene_enc, if_output_diffusion_process, start_data,reverse_steps,
                                    eval_mask, sampling, stride)
        
    def lat2traj(self, latent, V_k, s_mean, num_samples):
        return torch.matmul(latent, V_k.unsqueeze(0).repeat(num_samples,1,1)) + s_mean.unsqueeze(0)
    
            
    def task_diff(self, task, traj, label):
        if 'goal' in task:
            goal_pt = label
            if 'goal_at5s' in task:
                index = 50
            else:
                index = 59
                
            goal_diff = ((traj[:,:,index,:] - goal_pt.unsqueeze(1))**2)

            goal_diff[...,1] = goal_diff[...,1] * 1
            goal_diff = goal_diff.mean(-1)
            return goal_diff
            
        if 'map' in task:
            data = label
            map_point_pos = data['map_point']['position'] 
                        
            eval_mask = data['agent']['mask']
            traj = traj.squeeze(1)
            agent_batch = data['agent']['batch'][eval_mask] 
            map_min = data['map_min'].view(-1, 3)[..., :2][agent_batch]
            map_max = data['map_max'].view(-1, 3)[..., :2][agent_batch]
            
            traj = (traj + 1) * (map_max - map_min) / 2 + map_min
            edge_index_a2m = radius(
                x=map_point_pos[:, :2],
                y=traj[:, :2],
                r=torch.inf,
                batch_x=data['map_point']['batch'] if isinstance(data, Batch) else None,
                batch_y=data['agent']['batch'][eval_mask] if isinstance(data, Batch) else None,
                max_num_neighbors=5)
            rel_pos_a2m = traj[edge_index_a2m[0]] - map_point_pos[edge_index_a2m[1], :2]
            dist = torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1)
            min_dist = torch.zeros(traj.size(0), device=traj.device)
            for i in range(traj.size(0)):
                idx = torch.where(edge_index_a2m[0] == i)[0]
                min_dist[i] = dist[idx].min()
            
            goal_diff = min_dist

            # goal_diff[...,1] = goal_diff[...,1] * 1
            goal_diff = goal_diff.mean(-1)
            return goal_diff
        
        if 'original' in task:
            data = label

            eval_mask = data['agent']['mask']
            pos = traj.squeeze(1)[..., :2]
            agent_batch = data['agent']['batch'][eval_mask] 

            map_min = data['map_min'].view(-1, 3)[..., :2][agent_batch]
            map_max = data['map_max'].view(-1, 3)[..., :2][agent_batch]
            original_agent_position = data['agent']['position'][eval_mask, self.net.init_timestep-1, :2]
            
            
            pos = (pos + 1) * (map_max - map_min) / 2 + map_min

            min_dist = torch.norm(pos - original_agent_position, p=2, dim=-1)
            
            angle_diff =  traj.squeeze(1)[..., 2:4]
            heading = data['agent']['heading'][eval_mask, self.net.init_timestep-1].unsqueeze(-1)
            heading_vector = torch.cat([torch.cos(heading), torch.sin(heading)], dim=-1)
            angle_diff = angle_diff - heading_vector
            angle_diff =torch.norm(angle_diff, p=2, dim=-1)

            speed_diff = torch.norm(traj.squeeze(1)[..., 4] - data['agent']['init_speed'][eval_mask].unsqueeze(-1), p=2, dim=-1)
            goal_diff = torch.stack([min_dist, angle_diff, speed_diff], dim=-1)
            
            goal_diff = min_dist

            goal_diff = goal_diff.mean(-1)
            return goal_diff
    
    def sample_vd(self, 
               num_samples: int,
               data: HeteroData,
               scene_enc: Mapping[str, torch.Tensor],
               if_output_diffusion_process = False,
               start_data = None,
               reverse_steps = None,
               eval_mask = None,
               sampling="ddpm", 
               stride=20,
               ) -> Dict[str, torch.Tensor]:
        
        if reverse_steps is None:
            reverse_steps = self.var_sched.num_steps
        
        device = scene_enc['x_pt'].device

        num_agents = eval_mask.sum()
        
        e_init_rand = torch.randn([num_agents, num_samples, 5]).to(device)

        if start_data == None:
            x_init_T = e_init_rand
        
        else:
            c0 = torch.sqrt(self.var_sched.alpha_bars[reverse_steps]).to(device)
            c1 = torch.sqrt(1-self.var_sched.alpha_bars[reverse_steps]).to(device)
            x_init_T = c0 * start_data.unsqueeze(1) + c1 * e_init_rand
            
        x_init_t_list = [x_init_T]
        torch.cuda.empty_cache()
        
        for t in range(reverse_steps, 0, -stride):
            z_init = torch.randn_like(x_init_T) if t > 1 else torch.zeros_like(x_init_T)

            beta = self.var_sched.betas[t]
            
            alpha = self.var_sched.alphas[t]    
            alpha_bar = self.var_sched.alpha_bars[t]
            alpha_bar_next = self.var_sched.alpha_bars[t-stride]
            c0 = 1 / torch.sqrt(alpha)
            c1 = (1-alpha) / torch.sqrt(1 - alpha_bar)
            sigma = self.var_sched.get_sigmas(t, 0)
            
            x_init_t = x_init_t_list[-1]
            
            with torch.no_grad():
                beta = beta.unsqueeze(-1).repeat(num_agents * num_samples, 1).to(device)
                g_init_theta = self.net(copy.deepcopy(x_init_t), beta, data, scene_enc, num_samples = num_samples, eval_mask=eval_mask, mode=1)      

            if sampling == 'ddpm':
                x_init_next = c0 * (x_init_t - c1 * g_init_theta) + sigma * z_init
                
            elif sampling == 'ddim':

                x0_init_t = (x_init_t - g_init_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                x_init_next = alpha_bar_next.sqrt() * x0_init_t + (1 - alpha_bar_next).sqrt() * g_init_theta
                

            if True in torch.isnan(x_init_next):
                print('nan:',t)
            x_init_t_list.append(x_init_next.detach())
            if not if_output_diffusion_process:
                x_init_t_list.pop(0)
        
        
        if if_output_diffusion_process:
            return x_init_t_list
        else:
            return x_init_t_list[-1]

    
    def sample_guide(self, 
               num_samples: int,
               data: HeteroData,
               scene_enc: Mapping[str, torch.Tensor],
               if_output_diffusion_process = False,
               start_data = None,
               reverse_steps = None,
               eval_mask = None,
               sampling="ddpm", 
               stride=20,
               grad_guid = None,
               cond_gen = None,
               guid_param = None,
               uc = None,
               ) -> Dict[str, torch.Tensor]:
        
        if grad_guid != None:
            task = guid_param['task']
            guid_method = guid_param['guid_method']
            cost_param_costl = guid_param['cost_param']['cost_param_costl']
            cost_param_threl = guid_param['cost_param']['cost_param_threl']
            [guid_label, s_mean, V_k, VT_k, latent_mean, latent_std] = grad_guid
        
        if reverse_steps is None:
            reverse_steps = self.var_sched.num_steps
        
        device = scene_enc['x_pl'].device
        
        num_agents = eval_mask.sum()

        e_init_rand = torch.randn([num_agents, num_samples, 5]).to(device)
        
        
        s_T = torch.sqrt(self.var_sched.alpha_bars[reverse_steps].to(device))
        if start_data == None:
            c1 = 1
            x_init_T = c1 * e_init_rand + s_T
        else:
            c0 = torch.sqrt(self.var_sched.alpha_bars[reverse_steps]).to(device)
            c1 = torch.sqrt(1-self.var_sched.alpha_bars[reverse_steps]).to(device)

            if start_data.dim() == 2:
                x_init_T = c0 * start_data.unsqueeze(1) + c1 * e_init_rand
            elif start_data.dim() == 3:
                x_init_T = c0 * start_data + c1 * e_init_rand
        
        x_init_t_list = [x_init_T]
        
        torch.cuda.empty_cache()
        
        for t in range(reverse_steps, 0, -stride):

            z_init = torch.randn_like(x_init_T) if t > 1 else torch.zeros_like(x_init_T)
            beta = self.var_sched.betas[t]
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            alpha_bar_next = self.var_sched.alpha_bars[t-stride]
            c0 = 1 / torch.sqrt(alpha)
            c1 = (1-alpha) / torch.sqrt(1 - alpha_bar)
            
            
            x_init_t = x_init_t_list[-1]
            
            
            if cond_gen:
                [idx, target_mode] = cond_gen
                x_init_t[idx,:,:] = target_mode.unsqueeze(0).repeat(num_samples,1)
            clean_data = copy.deepcopy(x_init_T.squeeze(1)) # not used anyway for translation
            if guid_method == 'ECMR':
                with torch.no_grad():
                    beta_emb = beta.to(device).repeat(num_agents * num_samples).unsqueeze(-1)
                    
                    g_init_theta = self.net(copy.deepcopy(x_init_t), beta_emb, data, scene_enc, num_samples = num_samples, eval_mask=eval_mask, clean_data = clean_data)
                       
                with torch.inference_mode(False):
                    ### Marginal Mapping
                    temp_e = g_init_theta.clone().detach().float()
                    temp_x_t = x_init_t.clone().detach().float()
                    temp_x_0 = (temp_x_t - torch.sqrt(1-alpha_bar).clone() * temp_e) / torch.sqrt(alpha_bar).clone()
                    
                    ### Gradident.
                    temp_x_0.requires_grad = True
                    temp_x_0_unnor = temp_x_0 * (latent_std.unsqueeze(0).unsqueeze(0)+0.1) + latent_mean.unsqueeze(0).unsqueeze(0)
                    rec_traj = torch.matmul(temp_x_0_unnor.permute(1,0,2), V_k.unsqueeze(0).repeat(num_samples,1,1)) + s_mean.unsqueeze(0)
                    rec_traj = rec_traj.permute(1,0,2)
                    rec_traj = rec_traj.view(rec_traj.size(0), rec_traj.size(1),60,2)
                    diff = self.task_diff(task, rec_traj, guid_label)
                    error = diff.mean()
                    error.backward()
                    
                    # note it is the gradient of x0
                    grad = temp_x_0.grad
                    
                grad = grad * cost_param_costl
                scale = 1 * cost_param_threl
                grad = torch.clip(grad, min = -scale, max = scale)
                m_0 = temp_x_0 - grad
                                        
                x_init_next = alpha_bar_next.sqrt() * m_0 + (1 - alpha_bar_next).sqrt() * g_init_theta
                                    

            if guid_method == 'ECM':       
                    with torch.no_grad():
                        beta_emb = beta.to(device).repeat(num_agents * num_samples).unsqueeze(-1)
                        g_init_theta  = self.net(copy.deepcopy(x_init_t), beta_emb, data, scene_enc, num_samples = num_samples, eval_mask=eval_mask, clean_data = clean_data)
                        
                    with torch.inference_mode(False):
                        
                        temp_e = g_init_theta.clone().detach().float()
                        temp_x_t = x_init_t.clone().detach().float()
                        temp_x_0 = (temp_x_t - torch.sqrt(1-alpha_bar).clone() * temp_e) / torch.sqrt(alpha_bar).clone()
                        temp_x_0.requires_grad = True
                        recon_trans = temp_x_0
                        
                        diff = self.task_diff(task, recon_trans, guid_label)
                        error = diff.mean()
                        error.backward()
                        # note it is the gradient of x0
                        grad = temp_x_0.grad
                    grad = grad * cost_param_costl
                    scale = 1 * cost_param_threl
                    grad = torch.clip(grad, min = -scale, max = scale)
                    m_init_0 = temp_x_0 - grad
                    m_0 = torch.zeros_like(g_init_theta)
                
                    x_init_next = alpha_bar_next.sqrt() * m_init_0 + (1 - alpha_bar_next).sqrt() * g_init_theta
                    

            if True in torch.isnan(x_init_next):
                print('nan:',t)
            x_init_t_list.append(x_init_next.detach())
            
            if not if_output_diffusion_process:
                x_init_t_list.pop(0)
        
            
        if if_output_diffusion_process:
            return x_init_t_list
        else:
            return x_init_t_list[-1]

class InitDenoiser(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 init_timestep: int,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 diff_type: str,
                 m_dim: int) -> None:
        super(InitDenoiser, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.init_timestep = init_timestep
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.diff_type = diff_type
        self.m_dim = m_dim

        m_delta_dim = 5


        self.proj_in_m_delta = nn.Linear(m_delta_dim, self.hidden_dim)
        
        self.proj_in_m_delta_2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        
        self.proj_out_m_delta = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(self.hidden_dim, m_delta_dim),
        )

        
        
        noise_dim = 1
        self.noise_emb = FourierEmbedding(input_dim=noise_dim, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.type_a_emb = nn.Embedding(10, hidden_dim)
        ########
        
        self.interact_pt2m = nn.ModuleList(
            [TransformerDecoderLayerDiff(
            n_embd=hidden_dim,
            n_head=num_heads,
            ff_dim=4 * hidden_dim,
            dropout=dropout,
            layer_id=i,
        )  for i in range(num_layers)])
        
        self.to_out_m_delta = SkipMLP(d_model=hidden_dim)
        
        self.apply(weight_init)
    
    def forward(self,
                m_delta,
                beta,
                data: HeteroData,
                scene_enc: Mapping[str, torch.Tensor],
                num_samples: int,
                eval_mask,
                mode=0
     ) -> Dict[str, torch.Tensor]:

        device = m_delta.device
        
        self.num_samples = num_samples
        
        batch_size = data.num_graphs
        
        
        x_pt = scene_enc['x_pt'].repeat(self.num_samples, 1)
        x_pt_mask = data['map_point']['pt_mask']
        map_batch_list = data['map_point']['batch'][x_pt_mask]

        poly_cnt_per_batch = map_batch_list.bincount(minlength=batch_size)
        map_emb_batch = torch.split(x_pt, poly_cnt_per_batch.tolist())
        
        map_emb = pad_sequence(map_emb_batch, batch_first=True, padding_value=0) 
        
        
        beta_emb = self.noise_emb(beta)
                
        # num_agents x 128
        categorical_embs_m = [
                self.type_a_emb(data['agent']['type'][eval_mask].long()),
            ]
        
        m_delta = self.proj_in_m_delta(m_delta).view(-1, self.hidden_dim)
        m_delta = m_delta + categorical_embs_m[0]
        m_delta = self.proj_in_m_delta_2(m_delta)
        
        
        agent_batch_list = data['agent']['batch'][eval_mask]
        agent_cnt_per_batch = agent_batch_list.bincount(minlength=batch_size)
        agent_emb_batch = torch.split(m_delta, agent_cnt_per_batch.tolist())
        m_delta = pad_sequence(agent_emb_batch, batch_first=True, padding_value=0)
        pos_emb = sinusoidal_embedding(m_delta.shape[1], self.hidden_dim).to(device).unsqueeze(0)
        m_delta += pos_emb

        beta_emb_batch = torch.split(beta_emb, agent_cnt_per_batch.tolist())
        beta_emb_m = pad_sequence(beta_emb_batch, batch_first=True, padding_value=0)

        mask_map_layers = []
        mask_agent_layers = []

        attn_mask_map_layers = []
        attn_mask_agent_layers = []

        B, N, D = m_delta.shape
        B, N_map, _ = map_emb.shape
        
        for i in range(batch_size):
            mask_attn_map_agent_i = torch.arange(N).to(m_delta.device) < agent_cnt_per_batch[i]
            mask_attn_map_agent_i = mask_attn_map_agent_i.unsqueeze(-1).expand(-1, N_map)
            mask_attn_map_pt_i = torch.arange(N_map).to(m_delta.device) < poly_cnt_per_batch[i]
            attn_mask_map_layers.append(mask_attn_map_pt_i)
            mask_attn_map_pt_i = mask_attn_map_pt_i.unsqueeze(0).expand(N, -1)
            
            mask_attn_i = mask_attn_map_agent_i & mask_attn_map_pt_i
            mask_map_layers.append(mask_attn_i)
            
            
            mask_attn_agent_i = torch.arange(N).to(m_delta.device) < agent_cnt_per_batch[i]
            attn_mask_agent_layers.append(mask_attn_agent_i)
            mask_attn_agent_i = mask_attn_agent_i.unsqueeze(-1).expand(-1, N)
            mask_attn_i = mask_attn_agent_i & mask_attn_agent_i.t()
            mask_agent_layers.append(mask_attn_i)

        attn_mask_agent_layers = ~torch.stack(attn_mask_agent_layers)
        attn_mask_map_layers = ~torch.stack(attn_mask_map_layers)


        attn_mask_agent_layers = attn_mask_agent_layers.view(B, 1, N).to(torch.bool)
        attn_mask_map_layers = attn_mask_map_layers.view(B, 1, 1, N_map).   \
            expand(-1, self.num_heads*2, N, -1)
            
        # 0: don't attend others
        if mode == 0:
            attn_mask_agent_layers = attn_mask_agent_layers +  ~torch.eye(N).to(torch.bool).unsqueeze(0).to(m_delta.device)
 
        for i in range(self.num_layers):
            m_delta = m_delta + beta_emb_m
            m_delta = self.interact_pt2m[i](x=m_delta, map_enc=map_emb,
                                            mask=attn_mask_agent_layers, 
                                            map_mask=attn_mask_map_layers)


        mask = torch.arange(N).expand(B, N).to(m_delta.device) < agent_cnt_per_batch.unsqueeze(1)  # [B, N]
        mask_agent = mask.unsqueeze(-1).expand(-1, -1, D)  # [B, N, D]
        m_out_delta = m_delta[mask_agent].view(-1, D)  # [sum(agent_cnt_per_batch), D]

        out_m_delta = self.to_out_m_delta(m_out_delta)
        out_m_delta = out_m_delta.view(-1, self.num_samples, self.hidden_dim)
        
        return self.proj_out_m_delta(out_m_delta)


class VarianceSchedule(nn.Module):

    def __init__(self, num_steps, mode='linear',beta_1=1e-4, beta_T=5e-2,cosine_s=8e-3):
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            timesteps = (
            torch.arange(num_steps + 1) / num_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding
        
        alphas = 1 - betas
        
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()
        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)
        
        # kt
        sqrt_alpha_bars = torch.sqrt(alpha_bars)
        kt = 1 - sqrt_alpha_bars # shifted diffusion
        self.register_buffer('kt', kt)
        
        inv_sqrt_alpha = 1 / torch.sqrt(alphas)
        co_g = betas / torch.sqrt(1-alpha_bars)
        co_st = torch.sqrt(alphas[1:]) * (1-alpha_bars[:-1])/(1-alpha_bars[1:])
        co_st = torch.cat([torch.tensor([0]),co_st])
        co_z = torch.sqrt((1-alpha_bars[:-1])/(1-alpha_bars[1:])*betas[1:])
        co_z = torch.cat([torch.tensor([0]),co_z])
        self.register_buffer('inv_sqrt_alpha', inv_sqrt_alpha)
        self.register_buffer('co_g', co_g)
        self.register_buffer('co_st', co_st)
        self.register_buffer('co_z', co_z)
        

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas
    
class SkipMLP(torch.nn.Module):
    def __init__(self, d_model = 128, act_layer=nn.GELU):
        super().__init__()
        self.linear = torch.nn.Linear(d_model, d_model)
        self.ac = act_layer()
        self.norm1 = nn.LayerNorm([d_model])
        self.norm2 = nn.LayerNorm([d_model])

    def forward(self, x):          
        out = x + self.ac(self.linear(x))
        out = self.norm2(x + self.norm1(out))
        return out
