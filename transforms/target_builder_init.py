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
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
from utils import wrap_angle
from torch_cluster import radius


from torch_cluster import radius

class TargetBuilderInit(BaseTransform):

    def __init__(self,
                 init_timestep: int,
                 num_generation_timestep: int) -> None:
        self.init_timestep = init_timestep
        self.num_generation_timestep = num_generation_timestep
        
    def scale_given_map(self, data: HeteroData) -> HeteroData:
        mask = data['agent']['mask']
        
        map_size = torch.Tensor([250, 250, 2])
        map_size_half = map_size/2
        
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
        
        # Sort agents by their position and type
        points = data['agent']['position'][:, self.init_timestep, :2]
        _, indices_y = torch.sort(points[:, 1], stable=True)
        _, indices_x = torch.sort(points[indices_y, 0], stable=True)

        for key in data['agent'].keys():
            if 'ego' in key:
                continue
            if isinstance(data['agent'][key], torch.Tensor):
                
                data['agent'][key] = data['agent'][key][indices_y][indices_x]
                
            elif isinstance(data['agent'][key], list):
                data['agent'][key] = [data['agent'][key][indices_x[i]] for i in range(len(data['agent'][key]))]
            else:
                continue
        _, indices_type=torch.sort(data['agent']['type'], stable=True)
        
        for key in data['agent'].keys():
            if 'ego' in key:
                continue
            if isinstance(data['agent'][key], torch.Tensor):
                data['agent'][key] = data['agent'][key][indices_type]
                
            elif isinstance(data['agent'][key], list):
                data['agent'][key] = [data['agent'][key][indices_type[i]] for i in range(len(data['agent'][key]))]
            else:
                continue
            
        # Get the mask for the agents that are valid for prediction
        reg_mask = data['agent']['predict_mask'][:, self.init_timestep:]

        val_mask = data['agent']['valid_mask'][:, self.init_timestep]
        
        mask = (val_mask == True) & (reg_mask[:,0] == True ) & (reg_mask[:,-1] == True) &(data['agent']['type'] == 0)#((data['agent']['type'] == 0) | (data['agent']['type'] == 4))
        
        
        origin = data['agent']['position'][:, self.init_timestep]
        theta = data['agent']['heading'][:, self.init_timestep]
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(data['agent']['num_nodes'], 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos

        # Additional mask for agents that are in the lane

        side_pt_mask = data['map_point']['side'] != 2
        map_pts = data['map_point']['position']
        edge_a2m = radius(x = origin[..., :2], y = map_pts[side_pt_mask, :2], r = 2, max_num_neighbors = map_pts.size(0))

        center_pt_mask = data['map_point']['side'] == 2
        edge_a2m_center = radius(x = origin[..., :2], y = map_pts[center_pt_mask, :2], r = 4, max_num_neighbors = map_pts.size(0))

        in_lane_mask = torch.zeros_like(mask)
        in_lane_mask[edge_a2m[1].unique()] = True
        in_lane_mask[edge_a2m_center[1].unique()] = True
        
        data['agent']['mask'] = mask & in_lane_mask

        # Scale based on the location of the agent
        data = self.scale_given_map(data)
        init_translation = data['agent']['scaled_position'][:, self.init_timestep]
        init_velocity = data['agent']['velocity'][:, self.init_timestep, :]
        init_speed = torch.norm(init_velocity, p=2, dim=1)
        data['agent']['init_speed'] = init_speed / 15 - 1
        data['agent']['init_angle'] = theta

        data['agent']['init_translation'] = init_translation
        if init_translation.size()[0] != data['agent']['init_angle'].size()[0]:
            print("size error")

        # target: relative to its own origin
        data['agent']['target'] = origin.new_zeros(data['agent']['num_nodes'], self.num_generation_timestep, 4)
        data['agent']['target'][..., :2] = torch.bmm(data['agent']['position'][:, self.init_timestep:, :2] -
                                                     origin[:, :2].unsqueeze(1), rot_mat)
        if data['agent']['position'].size(2) == 3:
            data['agent']['target'][..., 2] = (data['agent']['position'][:, self.init_timestep:, 2] -
                                               origin[:, 2].unsqueeze(-1))
        data['agent']['target'][..., 3] = wrap_angle(data['agent']['heading'][:, self.init_timestep:] -
                                                     theta.unsqueeze(-1))

        return data
