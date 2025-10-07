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
from typing import Optional

import torch
from torchmetrics import Metric


from torch_cluster import radius, radius_graph

class OffRoadTraj(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(OffRoadTraj, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses


    def update(self,
               pred: torch.Tensor,
               agent_batch: torch.Tensor,
                map_pts) -> None:
        if len(pred.shape) == 4:
            pred = pred.squeeze(1)
        num_agents = pred.size(0)
        center_mask = map_pts['side'] == 2
        map_pt_batch = map_pts['batch'][center_mask]
        max_neighbor = map_pt_batch.bincount().max().item()
        map_pts_pos = map_pts['position'][center_mask, :2]
        agent_batch = agent_batch.view(-1, 1).repeat(1, pred.size(1)).view(-1)
        pred = pred.view(-1, 2)
        edge_a2m = radius(pred, map_pts_pos, r = 4,batch_x = agent_batch, batch_y = map_pt_batch, max_num_neighbors=max_neighbor)
        in_map_agent_idx = edge_a2m[1].unique()// 60
        in_map_mask = torch.zeros((num_agents,)).bool()
        in_map_mask[in_map_agent_idx.unique()] = True

        agent_in_cnt = in_map_mask.sum()
        self.sum += (num_agents - agent_in_cnt)
        self.count += num_agents
        

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


class CollisionTraj(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(CollisionTraj, self).__init__(**kwargs)
        self.add_state('num_collided', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor,
               agent_batch: Optional[torch.Tensor] = None,
        ) -> None:
        
        pred = pred.squeeze(1)
        pred = pred.permute(1, 0, 2).reshape(-1, 2)
        batch = torch.arange(60).unsqueeze(1).repeat(1, len(agent_batch)).to(pred.device).view(-1)
        edge_collision = radius_graph(pred, r=2, batch=batch, loop=False)
        self.num_collided += len(edge_collision[0].unique())
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.num_collided / self.count
    