# Copyright (c) 2025, Da Saem Lee. All rights reserved.
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

import torch
from torchmetrics import Metric

from torch_cluster import radius, radius_graph

class OffRoad(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(OffRoad, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses


    def update(self,
               pred: torch.Tensor,
               agent_batch: torch.Tensor,
                map_pts) -> None:
        if len(pred.shape) == 3:
            pred = pred.squeeze(1)
        center_mask = map_pts['side'] == 2
        map_pt_batch = map_pts['batch'][center_mask]
        max_neighbor = map_pt_batch.bincount().max().item()
        map_pts_pos = map_pts['position'][center_mask, :2]

        edge_a2m = radius(pred, map_pts_pos, r = 4,batch_x = agent_batch, batch_y = map_pt_batch, max_num_neighbors=max_neighbor)
        
        in_map_mask = torch.zeros_like(pred[:, 0]).bool()
        in_map_mask[edge_a2m[1]] = True
        num_agents = pred.size(0)
        agent_in_cnt = in_map_mask.sum()
        self.sum += (num_agents - agent_in_cnt)
        self.count += num_agents
        

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


class NearestEdge(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(NearestEdge, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses
        
    def update(self,
               pred: torch.Tensor,
               agent_batch: torch.Tensor,
               map_pts) -> None:
        if len(pred.shape) == 3:
            pred = pred.squeeze(1)
        num_agents = pred.size(0)

        center_mask = map_pts['side'] == 2
        map_pt_batch = map_pts['batch'][center_mask]

        max_neighbor = map_pt_batch.bincount().max().item()
        map_pts_pos = map_pts['position'][center_mask, :2]

        edge_a2m = radius(pred, map_pts_pos, r = torch.inf,batch_x = agent_batch, batch_y = map_pt_batch, max_num_neighbors=max_neighbor)
        
        rel_pos_a2m = map_pts_pos[edge_a2m[0]] - pred[edge_a2m[1]]
        dist = torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1)
        for batch_i in range(edge_a2m[1].max()+1):
            agent_i = edge_a2m[1] == batch_i
            if agent_i.sum() == 0:
                continue
            dist_agent = dist[agent_i]
            self.sum += dist_agent.min()
        self.count += num_agents
        

    def compute(self) -> torch.Tensor:
        return self.sum / self.count

class Collision(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(Collision, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses
        # ['vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus', 'static', 'background',
        #                      'construction', 'riderless_bicycle', 'unknown']

    def update(self,
               pred: torch.Tensor,
                agent_batch: torch.Tensor,
                agent_type: torch.Tensor) -> None:
        if len(pred.shape) == 3:
            pred = pred.squeeze(1)
        edge_a2a = radius_graph(pred, r=2, batch=agent_batch, max_num_neighbors=agent_batch.bincount().max().item())
        rel_pos = pred[edge_a2a[1]] - pred[edge_a2a[0]]
        dist = torch.norm(rel_pos[:, :2], p=2, dim=-1)
        for batch_i in range(pred.size(0)):
            agent_i = (edge_a2a[1] == batch_i) | (edge_a2a[0] == batch_i)
            if agent_i.sum() == 0:
                continue
            agent_types_1 = agent_type[edge_a2a[0][agent_i]]
            agent_types_2 = agent_type[edge_a2a[1][agent_i]]
            both_ped = (agent_types_1==1) & (agent_types_2==1)
            if (both_ped).sum():
                min_dist = dist[agent_i][both_ped].min()
                if min_dist < 1:
                    self.sum += 1
            else:
                self.sum += 1
        self.count += pred.size(0)
            
    def compute(self) -> torch.Tensor:
        return self.sum / self.count
