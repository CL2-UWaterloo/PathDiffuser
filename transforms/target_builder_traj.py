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
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
import numpy as np
from utils import wrap_angle
import matplotlib.pyplot as plt

from pathlib import Path


from utils import safe_list_index
from utils import side_to_directed_lineseg
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from torch_cluster import radius
# Lateral trajectory generator

from collections import defaultdict

from av2.geometry.interpolate import compute_midpoint_line
from av2.map.map_api import ArgoverseStaticMap
from av2.map.map_primitives import Polyline
from av2.utils.io import read_json_file


def generate_lateral_trajectory(d0, dT, T, t_vals, A):
    delta = dT - d0
    b = torch.hstack([
        delta,
        torch.zeros_like(delta),
        torch.zeros_like(delta)
    ])
    
    a3_5 = torch.linalg.solve(A, b.T)
    a3, a4, a5 = a3_5.unsqueeze(-1)
    return d0 + a3*t_vals**3 + a4*t_vals**4 + a5*t_vals**5

def interp1d(x, xp, fp):
    inds = torch.searchsorted(xp, x).clamp(max=fp.size(0) - 1)
    inds0 = (inds - 1).clamp(min=0)
    inds1 = inds

    x0 = xp[inds0]
    x1 = xp[inds1]
    f0 = fp[inds0]
    f1 = fp[inds1]

    weight = ((x - x0) / (x1 - x0).clamp(min=1e-5)).unsqueeze(-1)
    return (f1 - f0) * weight + f0

def frenet_to_cartesian(s_vals, d_vals, path_x, path_y, num_points=10):
    # Compute arc-length reference
    dx = path_x[1:] - path_x[:-1]
    dy = path_y[1:] - path_y[:-1]
    ds = torch.sqrt(dx**2 + dy**2)
    s_ref = torch.cat([torch.zeros(1), torch.cumsum(ds, dim=0)]) 
    
    # Compute path heading and normal
    theta = torch.atan2(dy, dx)
    n_x = -torch.sin(theta)
    n_y = torch.cos(theta)
    n_x = torch.cat([n_x, n_x[-1:]])
    n_y = torch.cat([n_y, n_y[-1:]])
    x_vals = interp1d(s_vals, s_ref, path_x.unsqueeze(-1)).squeeze(-1)
    y_vals = interp1d(s_vals, s_ref, path_y.unsqueeze(-1)).squeeze(-1)
    nx_vals = interp1d(s_vals, s_ref, n_x.unsqueeze(-1)).squeeze(-1)
    ny_vals = interp1d(s_vals, s_ref, n_y.unsqueeze(-1)).squeeze(-1)
    # # Apply lateral offsets
    x_traj = x_vals + d_vals * nx_vals
    y_traj = y_vals + d_vals * ny_vals
    # === Step 4: Resample to fixed number of points ===
    traj_dx = x_traj[1:] - x_traj[:-1]
    traj_dy = y_traj[1:] - y_traj[:-1]
    traj_ds = torch.sqrt(traj_dx**2 + traj_dy**2)
    traj_arc_len = torch.cat([torch.zeros((1,)), torch.cumsum(traj_ds, dim=-1)])  # [L]
    new_s = torch.linspace(0, traj_arc_len[-1], num_points)
    if x_traj.shape[0] ==0:
        x_fixed = torch.zeros_like(new_s)
        y_fixed = torch.zeros_like(new_s)
    else:
        x_fixed = interp1d(new_s, traj_arc_len, x_traj.unsqueeze(-1)).squeeze(-1)
        y_fixed = interp1d(new_s, traj_arc_len, y_traj.unsqueeze(-1)).squeeze(-1)

    return x_fixed, y_fixed


def cartesian_to_frenet(point, path_x, path_y, dx, dy, min_idx):

    # Use forward or backward segment
    use_last = (min_idx >= len(path_x) - 1)
    seg_idx = torch.where(use_last, min_idx - 1, min_idx)
    seg_idx = seg_idx.clamp(0, len(path_x) - 2)

    # Segment direction and normal
    seg_dx = path_x[seg_idx + 1] - path_x[seg_idx]
    seg_dy = path_y[seg_idx + 1] - path_y[seg_idx]
    seg_len = torch.sqrt(seg_dx**2 + seg_dy**2)
    tangent = torch.stack([seg_dx, seg_dy], dim=-1) / seg_len.unsqueeze(-1)
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)  # rotate 90°

    # Compute relative offset
    base = torch.stack([path_x[seg_idx], path_y[seg_idx]], dim=-1)
    delta = point - base
    d = (delta * normal).sum(dim=-1)

    # Compute arc-length (s)
    seg_ds = torch.sqrt(dx**2 + dy**2)
    s_ref = torch.cat([torch.zeros(1), torch.cumsum(seg_ds, dim=0)])
    s = s_ref[seg_idx] + (delta * tangent).sum(dim=-1)

    return s, d


def dfs_paths(node, path, paths, pt2pl, pts, graph, pt_orientation, pl_orientation, max_size=None):
    if len(path) != 0:
        last_node = path[-1]
        pl1_pt_idx = pt2pl[0][pt2pl[1] == node]
        pl2_pt_idx = pt2pl[0][pt2pl[1] == last_node]
        pt_idx1 = torch.stack([pl1_pt_idx[0], pl1_pt_idx[-1]])
        pt_idx2 = torch.stack([pl2_pt_idx[0], pl2_pt_idx[-1]])
        dist = torch.cdist(pts[pt_idx1], pts[pt_idx2])
        
        if dist.min() > 3:
            return
        
        # if loop, save and return
        if node in path:
            paths.append(path.copy())
            return
        path.append(node)
    else:
        path.append(node)
        
    if max_size and len(path) >= max_size:
        paths.append(path.copy())
        path.pop()
        return
    if node not in graph or not graph[node]:  # leaf node
        paths.append(path.copy())
    else:
        for neighbor in graph[node]:
            dfs_paths(neighbor, path, paths, pt2pl, pts, graph,pt_orientation, pl_orientation,max_size)
    path.pop() 
    
def merge_pls(pl2pl, data):
    edge_pl2pl = pl2pl['edge_index']
    type_pl2pl = pl2pl['type']
    # step 1: build the graph
    graph = defaultdict(list)
    edge_filter = type_pl2pl == 1
    pl_type_filter = torch.where(data['map_polygon']['type'] == 0)[0]
    pl_type_filter0 = torch.isin(edge_pl2pl[0], pl_type_filter) & torch.isin(edge_pl2pl[1], pl_type_filter)
    edge_pl2pl_list = edge_pl2pl[:, edge_filter&pl_type_filter0].T.tolist()
    for u, v in edge_pl2pl_list:
        graph[u].append(v)
        
        
    pt2pl = data['map_point', 'to', 'map_polygon']['edge_index']
    pt_center_mask = torch.where(data['map_point']['side'] == 2)[0]
    pt2pl = pt2pl[:, torch.isin(pt2pl[0], pt_center_mask)]
    pts = data['map_point']['position']
    pt_orientation = data['map_point']['orientation']
    pl_orientation = data['map_polygon']['orientation']
    # step 2: DFS to explore and record paths
    paths = []
    max_size = 10
    # start DFS from all unique starting nodes (pl2pl[0])
    for node in graph.keys():
        dfs_paths(node, [], paths, pt2pl, pts, graph, pt_orientation, pl_orientation, max_size)
    merged_polygon_pts = []
    merged_polygon_pt_idx = []
    
    
    for component in paths:
    
        points_pos = [pts[pt2pl[0][pt2pl[1] == idx]] for idx in component]

        points_pos = torch.cat(points_pos, dim=0)
        merged_polygon_pts.append(points_pos)  # [num_points, 2]

        
    return paths, merged_polygon_pts, merged_polygon_pt_idx

def generate_frenet_trajectory_candidates(
    A: torch.Tensor,
    d0: torch.Tensor,
    v_list: torch.Tensor,
    dT_list: torch.Tensor,
    T: float,
    t_vals: torch.Tensor,
    s0: float,
    path_x: torch.Tensor,
    path_y: torch.Tensor,
    path_length: float = None,
    cand_dim: int=10
):

    device = t_vals.device
    d0 = d0.to(device)

    # create a grid of (v0, dT) combinations
    v_grid, dT_grid = torch.meshgrid(v_list, dT_list, indexing='ij')  # shape [V, D]
    v0_flat = v_grid.reshape(-1).unsqueeze(-1)  # [N, 1]
    dT_flat = dT_grid.reshape(-1).unsqueeze(-1) + d0 # [N, 1]
    v0_flat = torch.cat([torch.zeros(1, 1).to(device), v0_flat], dim=0)  # [N+1, 1]
    dT_flat = torch.cat([torch.ones(1, 1).to(device)*d0, dT_flat], dim=0)  # [N+1, 1]
    # step 1: generate s_vals for each candidate
    s_vals = s0 + v0_flat * t_vals.unsqueeze(0)  # [N, T]

    # create a boolean mask
    mask =torch.where(s_vals[:, -1] < path_length)[0]  # shape: [batch_size, sequence_length]

    s_vals = s_vals[mask]  
    
    # # step 2: generate d_vals using lateral quintic polynomial
    d_vals = generate_lateral_trajectory(d0=d0, dT=dT_flat, T=T, t_vals=t_vals, A=A)  # [N, T]
    
    # # step 3: convert to Cartesian coordinates
    candidates = []
    for i in range(s_vals.size(0)):

        x_fixed, y_fixed = frenet_to_cartesian(
            s_vals[i], d_vals[i], path_x, path_y, num_points=cand_dim
        )
        traj = torch.stack([x_fixed, y_fixed], dim=-1)  # [num_points, 2]
        candidates.append(traj)

    return torch.stack(candidates)  # [N, num_points, 2]

class TargetBuilderTraj(BaseTransform):

    def __init__(self,
                 init_timestep: int,
                 num_generation_timestep: int) -> None:
        self.init_timestep = init_timestep
        self.num_generation_timestep = num_generation_timestep

        self.d_list = []
        self.s_list = []
        self.num_cand = 10
        T = 6.0 # time horizon
        self.A = torch.tensor([
                [T**3, T**4, T**5],
                [3*T**2, 4*T**3, 5*T**4],
                [6*T, 12*T**2, 20*T**3]
            ], dtype=torch.float32)
        self.A_inv = torch.tensor(np.linalg.inv(self.A))
         
        self.init_vel_list = torch.linspace(5, 30, 7)
        self.d_targets_list = torch.tensor([-2, -1.5, -1, 0, 1, 1.5, 2], dtype=torch.float32)
        
        self.cand_dim = 60
        
    def scale_given_map(self, data: HeteroData) -> HeteroData:
        mask = data['agent']['mask']
        
        map_size = torch.Tensor([250, 250, 2])
        map_size_half = map_size/2
        if mask.sum() == 0:
            mask = torch.ones_like(mask).bool()
        agent_locations = data['agent']['position'][mask, self.init_timestep].view(-1, 3)
        map_mid = (agent_locations.max(dim=0)[0] + agent_locations.min(dim=0)[0]) / 2
        
        map_min = map_mid - map_size_half
        map_max = map_mid + map_size_half

        data['map_min'] = map_min
        data['map_max'] = map_max
        positions = data['agent']['position']
        
        positions = 2* (positions - map_min) / (map_max - map_min) - 1

        data['agent']['scaled_position'] = positions[..., :2]

        return data

    def __call__(self, data: HeteroData) -> HeteroData:
        
        reg_mask = data['agent']['predict_mask'][:, self.init_timestep:]
        mask = (data['agent']['category'] >= 2) & (reg_mask[:,-1] == True) & (reg_mask[:,0] == True) & ((data['agent']['type']==0) | (data['agent']['type']==2) | (data['agent']['type']==4))

        data['agent']['mask'] = mask

        origin = data['agent']['position'][:, self.init_timestep]
        theta = data['agent']['heading'][:, self.init_timestep]

        # Scale based on the location of the agent
        data = self.scale_given_map(data)
        init_translation = data['agent']['scaled_position'][:, self.init_timestep]# - data['agent']['scaled_position'][ego_idx, self.init_timestep]
        init_velocity = data['agent']['velocity'][:, self.init_timestep, :]
        init_speed = torch.norm(init_velocity, p=2, dim=1)
        data['agent']['init_speed'] = init_speed / 15 - 1
        data['agent']['init_angle'] = theta

        data['agent']['init_translation'] = init_translation
        if init_translation.size()[0] != data['agent']['init_angle'].size()[0]:
            print("size error")
        ###############
        origin = data['agent']['position'][:, self.init_timestep]
        theta = data['agent']['heading'][:, self.init_timestep]
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(data['agent']['num_nodes'], 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos

        # target: relative to its own origin
        data['agent']['target'] = origin.new_zeros(data['agent']['num_nodes'], self.num_generation_timestep, 4)
        data['agent']['target'][..., :2] = torch.bmm(data['agent']['position'][:, self.init_timestep:, :2] -
                                                     origin[:, :2].unsqueeze(1), rot_mat)
        if data['agent']['position'].size(2) == 3:
            data['agent']['target'][..., 2] = (data['agent']['position'][:, self.init_timestep:, 2] -
                                               origin[:, 2].unsqueeze(-1))
        data['agent']['target'][..., 3] = wrap_angle(data['agent']['heading'][:, self.init_timestep:] -
                                                     theta.unsqueeze(-1))
        
        pt2pl = data['map_point', 'to', 'map_polygon']['edge_index']
        pl2pl = data['map_polygon', 'to', 'map_polygon']

        merged_polygons, merged_polygon_pts, merged_polygon_pt_idx = merge_pls(pl2pl, data)

        
        # Trajectory parameters
        T = 6.0 # time horizon
        t_vals = torch.linspace(0, T, self.cand_dim) # time interval.
        self.A = self.A.to(theta.device)

        starting_pos = data['agent']['position'][mask, self.init_timestep, :2]
        starting_head = data['agent']['heading'][mask, self.init_timestep]
        

        # filter out non-center line points and corresponding polygons
        pt_center_filter = torch.isin(pt2pl[0], torch.where(data['map_point']['side']==2)[0])
        
        # filter out polygons that are not for vehicle
        pl_mask = torch.where(data['map_polygon']['type'] == 0)[0]
        pt2pl = pt2pl[:, torch.isin(pt2pl[1], pl_mask) & pt_center_filter]
        
        # need to be center point of the vehicle lane
        edge_a2pt = radius(starting_pos, data['map_point']['position'][:, :2], r=5, max_num_neighbors=100)
        # filter out points that are not the center points and not the vehicle lane points
        edge_filter = torch.isin(edge_a2pt[0], pt2pl[0])
        edge_a2pt = edge_a2pt[:, edge_filter]
        
        rel_pos = starting_pos[edge_a2pt[1]] - data['map_point']['position'][edge_a2pt[0], :2]
        dist = torch.norm(rel_pos, dim=1)
        rel_angle = wrap_angle(starting_head[edge_a2pt[1]] - data['map_point']['orientation'][edge_a2pt[0]])
        
        candidate_agent_index = []
        candidate_tensors = torch.zeros((0, self.cand_dim, 2), dtype=torch.float32)
        plot = False
        for agent_i in edge_a2pt[1].unique().tolist():
            
            if plot:
                plt.figure()
            mask_agent_i = edge_a2pt[1] == agent_i
            if mask_agent_i.sum() == 0 or data['agent']['type'][mask][agent_i].item() not in [0, 2, 4]:
                continue
            
            rel_angle_i = rel_angle[mask_agent_i]
            dist_i = dist[mask_agent_i]
            pts_i = edge_a2pt[0][mask_agent_i]
            # find pts that are close + aligned
            dist_topk = torch.where(dist_i < 3)[0]
            angle_threshold = torch.where(torch.abs(rel_angle_i) < 0.1)[0]
            closest_pt_idx = dist_topk[torch.isin(dist_topk, angle_threshold)]
            # find the closest points
            pt_closest_idx = pts_i[closest_pt_idx].unique()
            pt_closest_pos = data['map_point']['position'][pt_closest_idx]
            # find the closest polygon
            pl_i_list = pt2pl[1][torch.isin(pt2pl[0], pt_closest_idx)].unique()
            if len(pl_i_list) == 0:
                continue
            pl2pl = data['map_polygon', 'to', 'map_polygon']['edge_index']
            
            candidate_path_pl = []
            candidate_path_pts = []
            
            for path_idx, path in enumerate(merged_polygons):
                
                if path[0] in pl_i_list:
                    candidate_path_pl.append(path)
                    candidate_path_pts.append(merged_polygon_pts[path_idx])
        
            candidate_list_pl = []
            for path_xy in candidate_path_pts:
                
                path_x = path_xy[:, 0]
                path_y = path_xy[:, 1]
                # Compute heading and normal vectors
                if path_x.shape[0] < 2:
                    
                    continue
                
                dx = torch.gradient(path_x)[0]
                dy = torch.gradient(path_y)[0]
                arc_len = torch.sqrt(dx**2 + dy**2)
                s_ref = torch.cat([torch.zeros(1), torch.cumsum(arc_len, dim=0)])

                path_length = s_ref[-1]

                # Plotting
                if plot:
                    plt.plot(path_x.numpy(), path_y.numpy(), 'k--', label='Reference Path')
                
                
                min_idx = torch.where(torch.isin(path_xy, pt_closest_pos)[:, :2].sum(-1))[0][0]

                s0, d0 = cartesian_to_frenet(starting_pos[agent_i], path_x, path_y, dx, dy, min_idx)

                if s0 > path_length or s0 < -5:
                    candidate_list_pl.append(torch.zeros((self.num_cand, self.cand_dim, 2), dtype=torch.float32))
                    continue

                traj = generate_frenet_trajectory_candidates(self.A, d0, self.init_vel_list, self.d_targets_list, T, t_vals, s0, path_x, path_y, path_length, self.cand_dim)
               
                if traj.shape[0]== 0:
                    continue
                
                candidate_list_pl.append(traj)
            if len(candidate_list_pl) == 0:
                continue
            candidate_list_pl = torch.vstack(candidate_list_pl)       

            # check if the last point is in the map
            map_pts = data['map_point']['position']
            
            edge_c2pts_final = radius(map_pts[..., :2], candidate_list_pl[:, -1], r=3, max_num_neighbors=10)
            heading_cand = torch.atan2(candidate_list_pl[:, -1, 1] - candidate_list_pl[:, -2, 1],
                            candidate_list_pl[:, -1, 0] - candidate_list_pl[:, -2, 0])
            
            wrap_angle_final = wrap_angle(heading_cand[edge_c2pts_final[0]] - data['map_point']['orientation'][edge_c2pts_final[1]])
            final_angle_mask = edge_c2pts_final[0][torch.where(torch.abs(wrap_angle_final) < 0.5)[0]]
            candidate_list_pl = candidate_list_pl[final_angle_mask.unique()]
            
            traj_pts = candidate_list_pl.view(-1, 2)
            edge_c2pts = radius(map_pts[..., :2], traj_pts, r=3, max_num_neighbors=1)
            # filter if the final point is not in the map
            in_map_count = (edge_c2pts[0]// self.cand_dim).bincount()
            in_map_cand = in_map_count > 55
        
            candidate_list_pl = candidate_list_pl[in_map_cand]
            
            candidate_tensors = torch.cat([candidate_tensors, candidate_list_pl], dim=0)
            candidate_agent_index.extend([agent_i] * candidate_list_pl.shape[0])
            if plot:
                x_traj, y_traj = candidate_list_pl[..., :, 0], candidate_list_pl[..., :, 1]
                for traj_i in range(x_traj.shape[0]):
                    plt.plot(x_traj[traj_i].numpy(), y_traj[traj_i].numpy(), 'r--', markersize=3, label='Frenet Trajectory')
                plt.plot(x_traj[:,-1].numpy(), y_traj[:,-1].numpy(), 'mo', label='End')  # red dot

                plt.plot(x_traj[:, 0].numpy(), y_traj[:,0].numpy(), 'g.', label='Start')  # green dot
                
                plt.title('Frenet Trajectory Candidates')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.axis('equal')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'frenet_trajectory_candidates_{agent_i}.png')
                plt.close()
        # # # save the candidates
        heading_init = torch.atan2(candidate_tensors[:, 1, 1] - candidate_tensors[:, 0, 1],
                                candidate_tensors[:, 1, 0] - candidate_tensors[:, 0, 0])
        cos, sin = heading_init.cos(), heading_init.sin()
        rot_mat = heading_init.new_zeros(heading_init.size()[0], 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        
        rotated_candidates = torch.bmm(candidate_tensors - candidate_tensors[:, 0].unsqueeze(1),
                                       rot_mat)
        data['agent']['candidate_orientation'] = heading_init
        data['agent']['rotated_candidate_list'] = rotated_candidates
        data['agent']['candidate_list'] = candidate_tensors
        data['agent']['candidate_agent_index'] = torch.tensor(candidate_agent_index, dtype=torch.int64)
        data['agent']['num_candidates'] = torch.tensor(len(candidate_agent_index), dtype=torch.int64)

        
        return data



