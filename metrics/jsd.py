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
from typing import Optional

import torch
from torchmetrics import Metric

from metrics.utils import topk
from metrics.utils import valid_filter


from shapely import Polygon, Point

from pathlib import Path

from av2.utils.io import read_json_file
from glob import glob
from itertools import permutations

from torch_cluster import radius, radius_graph

from scipy.spatial.distance import jensenshannon
from utils.geometry import wrap_angle

from av2.map.map_primitives import Polyline

from av2.map.map_api import ArgoverseStaticMap

# Union-Find for grouping
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, u):
        while u != self.parent[u]:
            self.parent[u] = self.parent[self.parent[u]]
            u = self.parent[u]
        return u

    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu != pv:
            self.parent[pu] = pv

    def groups(self):
        clusters = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            clusters.setdefault(root, []).append(i)
        return list(clusters.values())

def get_intersection_centers(raw_file_name, raw_dir):
                
    
    map_dir = Path(raw_dir) / raw_file_name
    map_path = map_dir / sorted(map_dir.glob('log_map_archive_*.json'))[0]
    map_data = read_json_file(map_path)
    centerlines = {lane_segment['id']: Polyline.from_json_data(lane_segment['centerline'])
                    for lane_segment in map_data['lane_segments'].values()}
    map_api = ArgoverseStaticMap.from_json(map_path)
    lane_segment_ids = map_api.get_scenario_lane_segment_ids()
    cross_walk_ids = list(map_api.vector_pedestrian_crossings.keys())
    polygon_ids = lane_segment_ids + cross_walk_ids
    num_polygons = len(lane_segment_ids) + len(cross_walk_ids) * 2
    centerline_list = []
    intersection_list = []
    for lane_segment in map_api.get_scenario_lane_segments():
        centerline = torch.from_numpy(centerlines[lane_segment.id].xyz).float()
        centerline_list.append(centerline)
        intersection_list.append(lane_segment.is_intersection)
    intersection_list = torch.tensor(intersection_list)
    
    intersection_indices = torch.where(intersection_list)[0]
    N = len(intersection_indices)

    uf = UnionFind(N)
    
    threshold = 3
    merged_intersection = []
    for i in range(N):
        idx1 = intersection_indices[i]
        p1 = centerline_list[idx1][:, :2].unsqueeze(1)  # (L1, 1, 2)
        

        for j in range(i+1, N):

            idx2 = intersection_indices[j]
            p2 = centerline_list[idx2][:, :2].unsqueeze(0)# (1,L2, 2)

            pairwise_dist = torch.norm(p1 - p2, dim=-1)  # (L1, L2)
            min_dist = pairwise_dist.min()
            
            if min_dist < threshold:
                uf.union(i, j)
    
    groups = uf.groups()
    end_points = []
    merged_intersection = []
    heading_list = []
    for group in groups:
        end_point =  [centerline_list[intersection_indices[i]][[0, -1]] for i in group]
        headings = [(centerline_list[intersection_indices[i]][-1] - centerline_list[intersection_indices[i]][0]) / torch.norm(centerline_list[intersection_indices[i]][-1] - centerline_list[intersection_indices[i]][0]) for i in group]
        headings = [torch.atan2(heading[1], heading[0]) for heading in headings]
        center_lines = [centerline_list[intersection_indices[i]] for i in group]
        merged_intersection.append(torch.cat(center_lines, dim=0))
        end_points.append(torch.cat(end_point, dim=0))
        heading_list.append(torch.vstack(headings))
    # merged_intersection = [torch.cat([centerline_list[intersection_indices[i]] for i in group], dim=0) for group in groups]
    if len(merged_intersection) == 0:
        return torch.zeros(0, 2), torch.zeros(0, 2), torch.zeros(0, 2)
    center = []
    for intersection_polylines in merged_intersection:
        inter_min = intersection_polylines.min(dim=0)[0]
        inter_max = intersection_polylines.max(dim=0)[0]
        intersection_center = (inter_max + inter_min) / 2#intersection_polylines.mean(dim=0)
        center.append(intersection_center)
    center = torch.stack(center)
    end_points = torch.vstack(end_points)
    heading_list = torch.vstack(heading_list)

    return center, end_points, heading_list

    
class JSD_MAP_INTERSECTION(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(JSD_MAP_INTERSECTION, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses
        self.bin = 150
        self.hist_dist_a2map_pred = torch.zeros(self.bin)
        self.hist_angle_a2map_pred = torch.zeros(72)
        self.hist_dist_a2map_gt = torch.zeros(self.bin)
        self.hist_angle_a2map_gt = torch.zeros(72)

        mode = 'val'
        self.raw_dir = f"/home/ds3lee/Desktop/data/argo_datasets/{mode}/raw"
        self.center_list = []
        self.intersection_info = []

    def update(self,
               pred: torch.Tensor,
               gt: torch.Tensor,
               agent_batch: torch.Tensor,
                map_pts, data) -> None:
        if len(pred.shape) == 3:
            pred = pred.squeeze(1)
        
        map_pt_batch = map_pts['batch']
        max_neighbor = map_pt_batch.bincount().max().item()

        num_batch =map_pt_batch.max().item()+1

        # get intersection center
        intersection_center_batch = []
        intersection_centers= []
        intersection_end_pt = []
        intersection_end_pt_batch = []
        intersection_headings = []
        for batch_i in range(num_batch):
            center, end_points, headings = get_intersection_centers(data['scenario_id'][batch_i], self.raw_dir)
            if center.size(0) == 0:
                continue
            intersection_centers.append(center)
            intersection_center_batch.append(torch.ones((center.size(0),), dtype=torch.long) * batch_i)
            intersection_end_pt_batch.append(torch.ones((end_points.size(0),), dtype=torch.long) * batch_i)
            intersection_end_pt.append(end_points)
            intersection_headings.append(headings)

            
        map_intersection_center = torch.vstack(intersection_centers)[..., :2].to(device=pred.device)
        # print(len(intersection_end_pt), intersection_end_pt[0].shape)
        intersection_end_pt = torch.vstack(intersection_end_pt).to(device=pred.device)[..., :2]
        
        map_intersection_center_batch = torch.hstack(intersection_center_batch).to(device=pred.device)
        intersection_end_pt_batch = torch.hstack(intersection_end_pt_batch).to(device=pred.device)
        intersection_headings = torch.vstack(intersection_headings).to(device=pred.device).repeat(1, 2).view(-1)
        
        self.center_list = [map_intersection_center, map_intersection_center_batch,  intersection_end_pt, intersection_end_pt_batch, intersection_headings]
        # map_intersection_center = data['intersection_center']
        # got intersection center
        
        # get agents < 5 of the intersection center
        min_dist_a2edge_pred = torch.zeros_like(pred[:, 0])
        pred_pos = pred[:, :2]
        pred_angle = pred[:, 2]
        # edge_a2center = radius(pred_pos, map_intersection_center, r = 5, batch_x = agent_batch, batch_y = map_intersection_center_batch, max_num_neighbors=max_neighbor)
        
        edge_a2inter_edges = radius(pred_pos, intersection_end_pt, r = torch.inf, batch_x = agent_batch, batch_y = intersection_end_pt_batch, max_num_neighbors=max_neighbor)
    
            
        rel_pos = pred_pos[edge_a2inter_edges[1]] - intersection_end_pt[edge_a2inter_edges[0]]
        rel_angle = wrap_angle(pred_angle[edge_a2inter_edges[1]] - intersection_headings[edge_a2inter_edges[0]])
        dist_intersection2agent = torch.norm(rel_pos[:, :2], p=2, dim=-1)
        # get dist of the closest polygon from each polygon
        intersection2agent_dist = []
        intersection2agent_angle = []
        for i in range(intersection_end_pt.size(0)):
            agent_idx = torch.where(edge_a2inter_edges[0] == i)[0]
            if agent_idx.size(0) == 0:
                continue
            topk_dist = torch.topk(dist_intersection2agent[agent_idx], min(5, len(agent_idx)), largest=False)

            intersection2agent_dist.append(topk_dist.values)
            intersection2agent_angle.append(rel_angle[topk_dist.indices])

                
        dist_intersection2agent_pred = torch.hstack(intersection2agent_dist)
        angle_intersection2agent_pred = torch.hstack(intersection2agent_angle)
            
        
        if  self.hist_dist_a2map_pred.device != min_dist_a2edge_pred.device:
            self.hist_dist_a2map_pred = self.hist_dist_a2map_pred.to(min_dist_a2edge_pred.device)    
            self.hist_angle_a2map_pred = self.hist_angle_a2map_pred.to(angle_intersection2agent_pred.device)
        self.hist_dist_a2map_pred += torch.histc(dist_intersection2agent_pred, bins=self.bin, min=0, max=self.bin)
        self.hist_angle_a2map_pred += torch.histc(angle_intersection2agent_pred, bins=72, min=-3.14, max=3.14)
            
    
        gt_pos = gt[:, :2]
        gt_angle = gt[:, 2]

        # edge_a2center = radius(gt_pos, map_intersection_center, r = 5, batch_x = agent_batch, batch_y = map_intersection_center_batch, max_num_neighbors=max_neighbor)
        
        edge_a2inter_edges = radius(gt_pos, intersection_end_pt, r = torch.inf, batch_x = agent_batch, batch_y = intersection_end_pt_batch, max_num_neighbors=max_neighbor)

        rel_pos = pred_pos[edge_a2inter_edges[1]] - intersection_end_pt[edge_a2inter_edges[0]]
        rel_angle = wrap_angle(pred_angle[edge_a2inter_edges[1]] - intersection_headings[edge_a2inter_edges[0]])
        dist_intersection2agent = torch.norm(rel_pos[:, :2], p=2, dim=-1)
        # get dist of the closest polygon from each polygon
        intersection2agent_dist = []
        intersection2agent_angle = []
        for i in range(intersection_end_pt.size(0)):
            agent_idx = torch.where(edge_a2inter_edges[0] == i)[0]
            if agent_idx.size(0) == 0:
                continue
            topk_dist = torch.topk(dist_intersection2agent[agent_idx], min(5, len(agent_idx)), largest=False)

            intersection2agent_dist.append(topk_dist.values)
            intersection2agent_angle.append(rel_angle[topk_dist.indices])

                
        dist_intersection2agent_gt = torch.hstack(intersection2agent_dist)
        angle_intersection2agent_gt = torch.hstack(intersection2agent_angle)
            

        if  self.hist_dist_a2map_gt.device != dist_intersection2agent_gt.device:
            self.hist_dist_a2map_gt = self.hist_dist_a2map_gt.to(dist_intersection2agent_gt.device)    
            self.hist_angle_a2map_gt = self.hist_angle_a2map_gt.to(angle_intersection2agent_gt.device)
        self.hist_dist_a2map_gt += torch.histc(dist_intersection2agent_gt, bins=self.bin, min=0, max=self.bin)
        self.hist_angle_a2map_gt += torch.histc(angle_intersection2agent_gt, bins=72, min=-3.14, max=3.14)
        
        self.intersection_info = [dist_intersection2agent_pred, dist_intersection2agent_gt, angle_intersection2agent_pred, angle_intersection2agent_gt]

    
    def compute(self) -> torch.Tensor:
        # max_sum = max(self.hist_dist_a2map_pred.sum(), self.hist_dist_a2map_gt.sum())
        dist_cumulative_pred = self.hist_dist_a2map_pred / self.hist_dist_a2map_pred.sum()
        dist_cumulative_gt = self.hist_dist_a2map_gt / self.hist_dist_a2map_gt.sum()
        jsd_dist = jensenshannon(dist_cumulative_pred.detach().cpu().numpy(), dist_cumulative_gt.detach().cpu().numpy())
        
        return torch.Tensor([jsd_dist]).to(self.hist_dist_a2map_pred.device)
    def get_intersection_center(self):
        return self.center_list
    
    def get_angle_hist(self):
        return self.hist_angle_a2map_pred, self.hist_angle_a2map_gt
    def get_intersection_info(self):
        return self.intersection_info   
    
      
    


    
class JSD_MAP_INTERSECTION_FLOW_IN(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(JSD_MAP_INTERSECTION_FLOW_IN, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses
        self.radius = 100
        self.hist_dist_in_pred = torch.zeros(self.radius)
        self.hist_dist_out_pred = torch.zeros(self.radius)
        self.hist_dist_in_gt = torch.zeros(self.radius)
        self.hist_dist_out_gt = torch.zeros(self.radius)


    def update(self,
                intersection_info, device) -> None:
          
        # get intersection center
        dist_intersection2agent_pred, dist_intersection2agent_gt, angle_intersection2agent_pred, angle_intersection2agent_gt= intersection_info

        pred_close_agents = torch.where(dist_intersection2agent_pred < 5)[0]
        in_count_pred = 0
        out_count_pred = 0 
        if len(pred_close_agents) != 0:
            rel_angles = angle_intersection2agent_pred[pred_close_agents]
            in_agents = torch.where(torch.abs(rel_angles) < 0.1)[0]
            in_count_pred += in_agents.size(0)
            out_count_pred += len(pred_close_agents) - in_count_pred
        in_count_gt = 0
        out_count_gt = 0
        gt_close_agents = torch.where(dist_intersection2agent_gt < 5)[0]
        if len(gt_close_agents) != 0:
            rel_angles = angle_intersection2agent_gt[gt_close_agents]
            in_agents = torch.where(torch.abs(rel_angles) < 0.1)[0]
            in_count_gt += in_agents.size(0)
            out_count_gt += len(gt_close_agents) - in_count_pred

        if self.hist_dist_in_pred.device != device:
            self.hist_dist_in_pred = self.hist_dist_in_pred.to(device)
            self.hist_dist_out_pred = self.hist_dist_out_pred.to(device)
        in_count_pred = min(in_count_pred, self.hist_dist_in_pred.size(0)-1)
        out_count_pred = min(out_count_pred, self.hist_dist_in_pred.size(0)-1)
        self.hist_dist_in_pred[in_count_pred] += 1
        self.hist_dist_out_pred[out_count_pred] += 1
        
        
        if self.hist_dist_in_gt.device != device:
            self.hist_dist_in_gt = self.hist_dist_in_gt.to(device)
            self.hist_dist_out_gt = self.hist_dist_out_gt.to(device)
        in_count_gt = min(in_count_gt, self.hist_dist_in_gt.size(0)-1)
        out_count_gt = min(out_count_gt, self.hist_dist_in_gt.size(0)-1)
        self.hist_dist_in_gt[in_count_gt] += 1
        self.hist_dist_out_gt[out_count_gt] += 1
        
    
    def compute(self) -> torch.Tensor:
        in_cumulative_pred = self.hist_dist_in_pred / self.hist_dist_in_pred.sum()
        in_cumulative_gt = self.hist_dist_in_gt / self.hist_dist_in_gt.sum()
        
        in_jsd = jensenshannon(in_cumulative_pred.detach().cpu().numpy(), in_cumulative_gt.detach().cpu().numpy())

        return torch.Tensor([in_jsd]).to(self.hist_dist_out_gt.device)

    def get_out_jsd(self):
        return  self.hist_dist_out_pred,  self.hist_dist_out_gt
    

class JSD_MAP_INTERSECTION_FLOW_OUT(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(JSD_MAP_INTERSECTION_FLOW_OUT, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses
        self.radius = 100
        self.hist_dist_out_pred = torch.zeros(self.radius)
        self.hist_dist_out_gt = torch.zeros(self.radius)


    def update(self,
               hist_dist_out_pred: torch.Tensor,
               hist_dist_out_gt: torch.Tensor) -> None:
        self.hist_dist_out_pred = hist_dist_out_pred
        self.hist_dist_out_gt = hist_dist_out_gt
    
    def compute(self) -> torch.Tensor:
        out_cumulative_pred = self.hist_dist_out_pred / self.hist_dist_out_pred.sum()
        out_cumulative_gt = self.hist_dist_out_gt / self.hist_dist_out_gt.sum()
        out_jsd = jensenshannon(out_cumulative_pred.detach().cpu().numpy(), out_cumulative_gt.detach().cpu().numpy())
        return torch.Tensor([out_jsd]).to(self.hist_dist_out_gt.device)
    

class JSD_LOCAL_DENSITY(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(JSD_LOCAL_DENSITY, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses
        self.num_bin = 150
        self.local_density_pred = torch.zeros(self.num_bin)
        self.local_density_gt = torch.zeros(self.num_bin)
        


    def update(self,
               pred: torch.Tensor,
               gt: torch.Tensor,
               agent_batch: torch.Tensor,
                data) -> None:
        if len(pred.shape) == 3:
            pred = pred.squeeze(1)
        if self.local_density_pred.device != pred.device:
            self.local_density_pred = self.local_density_pred.to(pred.device)
            self.local_density_gt = self.local_density_gt.to(pred.device)

        edge_pred = radius_graph(pred, r = torch.inf, batch = agent_batch, max_num_neighbors=agent_batch.bincount().max())
        rel_pos = pred[edge_pred[1]] - pred[edge_pred[0]]
        dist = torch.norm(rel_pos[:, :2], p=2, dim=-1)
        top5_dist_list = []
        for agent_i in range(pred.size(0)):
            agent_idx = torch.where(edge_pred[1] == agent_i)[0]
            if agent_idx.size(0) == 0:
                continue
            min_dist = torch.topk(dist[agent_idx], min(5, len(agent_idx)), largest=False)
            top5_dist_list.append(min_dist.values)
        top5_dist_list = torch.cat(top5_dist_list)
        self.local_density_pred += torch.histc(top5_dist_list, bins=self.num_bin, min=0, max=self.num_bin)


        edge_gt = radius_graph(gt, r = torch.inf, batch = agent_batch, max_num_neighbors=agent_batch.bincount().max())
        rel_pos = gt[edge_gt[1]] - gt[edge_gt[0]]
        dist = torch.norm(rel_pos[:, :2], p=2, dim=-1)
        top5_dist_list = []
        for agent_i in range(pred.size(0)):
            agent_idx = torch.where(edge_pred[1] == agent_i)[0]
            if agent_idx.size(0) == 0:
                continue
            min_dist = torch.topk(dist[agent_idx], min(5, len(agent_idx)), largest=False)
            top5_dist_list.append(min_dist.values)
        top5_dist_list = torch.cat(top5_dist_list)

        self.local_density_gt += torch.histc(top5_dist_list, bins=self.num_bin, min=0, max=self.num_bin)


    def compute(self) -> torch.Tensor:
        
        dist_cumulative_pred = self.local_density_pred / self.local_density_pred.sum() 
        dist_cumulative_gt = self.local_density_gt / self.local_density_gt.sum()
        jsd_dist = jensenshannon(dist_cumulative_pred.detach().cpu().numpy(), dist_cumulative_gt.detach().cpu().numpy())
        
        return torch.Tensor([jsd_dist]).to(self.local_density_gt.device)


class JSD_MAP_INTERSECTION_ANGLE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(JSD_MAP_INTERSECTION_ANGLE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

        self.hist_angle_a2map_pred = torch.zeros(72)
        self.hist_angle_a2map_gt = torch.zeros(72)


    def update(self,
               angle_hist_gt, angle_hist_pred) -> None:
        
        if  self.hist_angle_a2map_pred.device != angle_hist_pred.device:
            self.hist_angle_a2map_pred = self.hist_angle_a2map_pred.to(angle_hist_pred.device)
        if  self.hist_angle_a2map_gt.device != angle_hist_gt.device:
            self.hist_angle_a2map_gt = self.hist_angle_a2map_gt.to(angle_hist_gt.device)
        
        self.hist_angle_a2map_pred += angle_hist_pred
        
        self.hist_angle_a2map_gt += angle_hist_gt

    def compute(self) -> torch.Tensor:
        angle_cumulative_pred = self.hist_angle_a2map_pred / self.hist_angle_a2map_pred.sum()
        angle_cumulative_gt = self.hist_angle_a2map_gt / self.hist_angle_a2map_gt.sum()
        jsd_angle = jensenshannon(angle_cumulative_pred.detach().cpu().numpy(), angle_cumulative_gt.detach().cpu().numpy())

        return torch.Tensor([jsd_angle]).to(self.hist_angle_a2map_pred.device)
    
class JSD_MAP_DIST(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(JSD_MAP_DIST, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

        self.hist_dist_a2map_pred = torch.zeros(30)
        self.hist_dist_a2map_gt = torch.zeros(30)


    def update(self,
               pred: torch.Tensor,
               gt: torch.Tensor,
               agent_batch: torch.Tensor,
                map_pts) -> None:
        if len(pred.shape) == 3:
            pred = pred.squeeze(1)
        center_mask = map_pts['side'] == 2
        map_pt_batch = map_pts['batch'][center_mask]
        max_neighbor = map_pt_batch.bincount().max().item()
        map_pts_pos = map_pts['position'][center_mask, :2]
        map_pts_orient = map_pts['orientation'][center_mask]
        
        min_dist_a2edge_pred = torch.zeros_like(pred[:, 0])
        min_angledev_a2edge_pred = torch.zeros_like(pred[:, 0])
        pred_pos = pred[:, :2]
        pred_angle = pred[:, 2]
        edge_a2m = radius(pred_pos, map_pts_pos, r = torch.inf, batch_x = agent_batch, batch_y = map_pt_batch, max_num_neighbors=max_neighbor)
        
        rel_pos = pred_pos[edge_a2m[1]] - map_pts_pos[edge_a2m[0]]
        rel_angle = wrap_angle(pred_angle[edge_a2m[1]] - map_pts_orient[edge_a2m[0]])
        dist = torch.norm(rel_pos[:, :2], p=2, dim=-1)
        for batch_i in range(pred_pos.size(0)):
            agent_i = (edge_a2m[1] == batch_i)
            if agent_i.sum() == 0:
                continue
            agent_idx = torch.where(agent_i)[0]
            min_dist_idx = agent_idx[dist[agent_i].argmin()]
            self.closest_map_idx_pred = min_dist_idx

            min_dist_a2edge_pred[batch_i] = dist[min_dist_idx]
            min_angledev_a2edge_pred[batch_i] = rel_angle[min_dist_idx]
        min_angledev_a2edge_pred = torch.rad2deg(min_angledev_a2edge_pred)
        if  self.hist_dist_a2map_pred.device != min_dist_a2edge_pred.device:
            self.hist_dist_a2map_pred = self.hist_dist_a2map_pred.to(min_dist_a2edge_pred.device)
            
        self.hist_dist_a2map_pred += torch.histc(min_dist_a2edge_pred, bins=30, min=0, max=3)
        self.hist_angle_a2map_pred = torch.histc(min_angledev_a2edge_pred, bins=72, min=-180, max=180)
        
        
        min_dist_a2edge_gt = torch.zeros_like(pred[:, 0])
        min_angledev_a2edge_gt = torch.zeros_like(pred[:, 0])
        gt_pos = gt[:, :2]
        gt_angle = gt[:, 2]
        edge_a2m = radius(gt_pos, map_pts_pos, r = torch.inf, batch_x = agent_batch, batch_y = map_pt_batch, max_num_neighbors=max_neighbor)
        
        rel_pos = gt_pos[edge_a2m[1]] - map_pts_pos[edge_a2m[0]]
        rel_angle = wrap_angle(gt_angle[edge_a2m[1]] - map_pts_orient[edge_a2m[0]])
        dist = torch.norm(rel_pos[:, :2], p=2, dim=-1)
        for batch_i in range(gt_pos.size(0)):
            agent_i = (edge_a2m[1] == batch_i)
            if agent_i.sum() == 0:
                continue
            agent_idx = torch.where(agent_i)[0]
            min_dist_idx = agent_idx[dist[agent_i].argmin()]
            self.closest_map_idx_gt = min_dist_idx
            min_dist_a2edge_gt[batch_i] = dist[min_dist_idx]
            min_angledev_a2edge_gt[batch_i] = rel_angle[min_dist_idx]
        min_angledev_a2edge_gt = torch.rad2deg(min_angledev_a2edge_gt)
        
        if self.hist_dist_a2map_gt.device != min_dist_a2edge_gt.device:
            self.hist_dist_a2map_gt = self.hist_dist_a2map_gt.to(min_dist_a2edge_gt.device)
            
        self.hist_dist_a2map_gt += torch.histc(min_dist_a2edge_gt, bins=30, min=0, max=3)
        self.hist_angle_a2map_gt = torch.histc(min_angledev_a2edge_gt, bins=72, min=-180, max=180)
        
        
    def get_angle_hist(self):
        return self.hist_angle_a2map_gt, self.hist_angle_a2map_pred
    
    def compute(self) -> torch.Tensor:
        dist_cumulative_pred = self.hist_dist_a2map_pred / self.hist_dist_a2map_pred.sum()
        dist_cumulative_gt = self.hist_dist_a2map_gt / self.hist_dist_a2map_gt.sum()
        jsd_dist = jensenshannon(dist_cumulative_pred.detach().cpu().numpy(), dist_cumulative_gt.detach().cpu().numpy())
        
        return torch.Tensor([jsd_dist]).to(self.hist_dist_a2map_pred.device)


class JSD_MAP_ANGLE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(JSD_MAP_ANGLE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

        self.hist_angle_a2map_pred = torch.zeros(72)
        self.hist_angle_a2map_gt = torch.zeros(72)


    def update(self,
               angle_hist_gt, angle_hist_pred) -> None:
        
        if  self.hist_angle_a2map_pred.device != angle_hist_pred.device:
            self.hist_angle_a2map_pred = self.hist_angle_a2map_pred.to(angle_hist_pred.device)
        if  self.hist_angle_a2map_gt.device != angle_hist_gt.device:
            self.hist_angle_a2map_gt = self.hist_angle_a2map_gt.to(angle_hist_gt.device)
        
        self.hist_angle_a2map_pred += angle_hist_pred
        
        self.hist_angle_a2map_gt += angle_hist_gt

    def compute(self) -> torch.Tensor:
        angle_cumulative_pred = self.hist_angle_a2map_pred / self.hist_angle_a2map_pred.sum()
        angle_cumulative_gt = self.hist_angle_a2map_gt / self.hist_angle_a2map_gt.sum()
        jsd_angle = jensenshannon(angle_cumulative_pred.detach().cpu().numpy(), angle_cumulative_gt.detach().cpu().numpy())

        return torch.Tensor([jsd_angle]).to(self.hist_angle_a2map_pred.device)

class JSD_INTERACTIVE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(JSD_INTERACTIVE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses
        
        self.hist_dist_a2a_pred = torch.zeros(250)
        self.hist_dist_a2a_gt = torch.zeros(250)

    def update(self,
               pred: torch.Tensor,
               gt: torch.Tensor,
               agent_batch: torch.Tensor,
               ) -> None:
        if len(pred.shape) == 3:
            pred = pred.squeeze(1)

        edge_a2a = radius_graph(pred, r=torch.inf, batch=agent_batch, max_num_neighbors=agent_batch.bincount().max().item())
        rel_pos = pred[edge_a2a[1]] - pred[edge_a2a[0]]
        dist = torch.norm(rel_pos[:, :2], p=2, dim=-1)

        min_dist_a2a_pred = torch.zeros_like(pred[:, 0])
        
        for batch_i in range(pred.size(0)):
            agent_i = (edge_a2a[1] == batch_i) | (edge_a2a[0] == batch_i)
            if agent_i.sum() == 0:
                continue

            min_dist_a2a_pred[batch_i] = dist[agent_i].min()

        if self.hist_dist_a2a_pred.device != min_dist_a2a_pred.device:
            self.hist_dist_a2a_pred = self.hist_dist_a2a_pred.to(min_dist_a2a_pred.device)
        self.hist_dist_a2a_pred += torch.histc(min_dist_a2a_pred, bins=250, min=0, max=250)


        edge_a2a = radius_graph(gt, r=torch.inf, batch=agent_batch, max_num_neighbors=agent_batch.bincount().max().item())
        rel_pos = gt[edge_a2a[1]] - gt[edge_a2a[0]]
        dist = torch.norm(rel_pos[:, :2], p=2, dim=-1)

        
        min_dist_a2a_gt = torch.zeros_like(gt[:, 0])
        for batch_i in range(gt.size(0)):
            agent_i = (edge_a2a[1] == batch_i) | (edge_a2a[0] == batch_i)
            if agent_i.sum() == 0:
                continue

            min_dist_a2a_gt[batch_i] = dist[agent_i].min()

        if self.hist_dist_a2a_gt.device != min_dist_a2a_gt.device:
            self.hist_dist_a2a_gt = self.hist_dist_a2a_gt.to(min_dist_a2a_gt.device)
        self.hist_dist_a2a_gt += torch.histc(min_dist_a2a_gt, bins=250, min=0, max=250)


    def compute(self) -> torch.Tensor:
        dist_cumulative_pred = self.hist_dist_a2a_pred / self.hist_dist_a2a_pred.sum()
        dist_cumulative_gt = self.hist_dist_a2a_gt / self.hist_dist_a2a_gt.sum()
        jsd_dist = jensenshannon(dist_cumulative_pred.detach().cpu().numpy(), dist_cumulative_gt.detach().cpu().numpy())
        return torch.Tensor([jsd_dist]).to(self.hist_dist_a2a_pred.device)

class JSD_SPEED(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(JSD_SPEED, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses
        # ['vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus', 'static', 'background',
        #                      'construction', 'riderless_bicycle', 'unknown']

        self.hist_speed_pred = torch.zeros(50)
        self.hist_speed_gt = torch.zeros(50)

    def update(self,
               pred: torch.Tensor,
               gt: torch.Tensor,
                ) -> None:
        if len(pred.shape) == 3:
            pred = pred.squeeze(1)
        pred = (pred + 1) * 15
        gt = (gt + 1) * 15
        if self.hist_speed_pred.device != pred.device:
            self.hist_speed_pred = self.hist_speed_pred.to(pred.device)
            self.hist_speed_gt = self.hist_speed_gt.to(pred.device)
        self.hist_speed_pred += torch.histc(pred, bins=50, min=0, max=50)
        self.hist_speed_gt += torch.histc(gt, bins=50, min=0, max=50)
        
            
    def compute(self) -> torch.Tensor:
        speed_cumulative_pred = self.hist_speed_pred / self.hist_speed_pred.sum()
        speed_cumulative_gt = self.hist_speed_gt / self.hist_speed_gt.sum()
        jsd_speed = jensenshannon(speed_cumulative_pred.detach().cpu().numpy(), speed_cumulative_gt.detach().cpu().numpy())
        return torch.Tensor([jsd_speed]).to(self.hist_speed_pred.device)