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
from typing import Dict

import torch
import torch.nn as nn
from torch_cluster import radius
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from layers.attention_layer import AttentionLayer
from layers.fourier_embedding import FourierEmbedding
from utils import angle_between_2d_vectors
from utils import weight_init
from utils import wrap_angle


class QCNetMapEncoderPT(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 init_timestep: int,
                 pl2pl_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        super(QCNetMapEncoderPT, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.init_timestep = init_timestep
        self.pl2pl_radius = pl2pl_radius
        # num_freq_bands = 64
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        if dataset == 'argoverse_v2':
            if input_dim == 2:
                input_dim_x_pt = 5
                input_dim_x_pl = 0
                input_dim_r_pt2pl = 3
                input_dim_r_pl2pl = 3
            elif input_dim == 3:
                input_dim_x_pt = 2
                input_dim_x_pl = 1
                input_dim_r_pt2pl = 4
                input_dim_r_pl2pl = 4
            else:
                raise ValueError('{} is not a valid dimension'.format(input_dim))
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))

        if dataset == 'argoverse_v2':
            self.type_pt_emb = nn.Embedding(17, hidden_dim)
            self.side_pt_emb = nn.Embedding(3, hidden_dim)
            self.type_pl_emb = nn.Embedding(4, hidden_dim)
            self.int_pl_emb = nn.Embedding(3, hidden_dim)
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))
        self.type_pt2pt_emb = nn.Embedding(5, hidden_dim)
        self.x_pt_emb = FourierEmbedding(input_dim=input_dim_x_pt, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.x_pl_emb = FourierEmbedding(input_dim=input_dim_x_pl, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        
        self.r_pt2pt_emb = FourierEmbedding(input_dim=input_dim_r_pt2pl, hidden_dim=hidden_dim,
                                            num_freq_bands=num_freq_bands)
        self.pt2pt_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.apply(weight_init)

    def compute_curvature(self, data):
        edge_index_pt2pl = data['map_point', 'to', 'map_polygon']['edge_index']
        
        pos = data['map_point']['position'][:, :2]
        x, y = pos[..., 0], pos[..., 1]
        dx = torch.gradient(x, dim=0)[0]  # dx/ds
        dy = torch.gradient(y, dim=0)[0]  # dy/ds

        # Compute second derivatives (d²x/ds² and d²y/ds²)
        ddx = torch.gradient(dx, dim=0)[0]  # d²x/ds²
        ddy = torch.gradient(dy, dim=0)[0]  # d²y/ds²

        # Compute curvature using the formula
        numerator = torch.abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2) ** (3/2)

        # Avoid division by zero
        curvature = numerator / (denominator + 1e-6)
        final_pt = torch.where(torch.gradient(edge_index_pt2pl[1], dim=0)[0] != 0)[0]
        curvature[final_pt] = curvature[final_pt-1]

        return curvature
        
    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
    
        pos_pt = data['map_point']['position'][:, :self.input_dim].contiguous()
        orient_pt = data['map_point']['orientation'].contiguous()
        orient_vector_pt = torch.stack([orient_pt.cos(), orient_pt.sin()], dim=-1)

        pos_pl = data['map_polygon']['position'][:, :self.input_dim].contiguous()
        orient_pl = data['map_polygon']['orientation'].contiguous()
        orient_vector_pl = torch.stack([orient_pl.cos(), orient_pl.sin()], dim=-1)

        curvatures = self.compute_curvature(data)

        pt_side_mask = data['map_point']['side'] == 2
        edge_index_pt2pl = data['map_point', 'to', 'map_polygon']['edge_index']
        pt_mask = torch.zeros_like(pt_side_mask, dtype=torch.bool)
        pt2pl_edge_mask = torch.isin(edge_index_pt2pl[0], torch.where(pt_side_mask)[0])
        edge_index_pt2pl = edge_index_pt2pl[:, pt2pl_edge_mask]

        for pl_i in range(edge_index_pt2pl[1].max()):
            pl_mask = edge_index_pt2pl[1] == pl_i
            pl2pt_mask = edge_index_pt2pl[0][pl_mask]
            dist_mask = torch.where(pl2pt_mask)[0][::2]
            pt_mask[pl2pt_mask[dist_mask]] = True
        
        data['map_point']['pt_mask'] =pt_mask

        pt_mask_idx = torch.where(pt_mask)[0]

        edge_index_pt2pl = data['map_point', 'to', 'map_polygon']['edge_index']
        pt2pl_edge_mask = torch.isin(edge_index_pt2pl[0], pt_mask_idx)
        edge_index_pt2pl = edge_index_pt2pl[:, pt2pl_edge_mask]
        edge_index_pt2pl[0] = torch.arange(edge_index_pt2pl[0].size(0))
        edge_index_pt2pt = []
        end_pts = []
        for pl_idx in range(edge_index_pt2pl[1].max()):
            pl_mask = edge_index_pt2pl[1] == pl_idx
            pl2pt_mask = edge_index_pt2pl[0][pl_mask]
            pt2pt_edge1 = torch.stack([pl2pt_mask[:-1], pl2pt_mask[1:]])
            pt2pt_edge2 = torch.stack([pl2pt_mask[1:], pl2pt_mask[:-1]])
            pt2pt_edge = torch.cat([pt2pt_edge1, pt2pt_edge2], dim=-1)
            edge_index_pt2pt.append(pt2pt_edge)
            end_pts.append(pl2pt_mask[-1])
            end_pts.append(pl2pt_mask[0])
        edge_index_pt2pt = torch.cat(edge_index_pt2pt, dim=-1)
        end_pts = torch.stack(end_pts).unique()
        pl_is_intersection = torch.where(data['map_polygon']['is_intersection'] == True)[0]
        pl_not_intersection = torch.where(data['map_polygon']['is_intersection'] != True)[0]
        pt_intersection_mask = torch.isin(edge_index_pt2pl[0], pl_is_intersection)
        pt_not_intersection_mask = torch.isin(edge_index_pt2pl[0], pl_not_intersection)

        pos_pt = pos_pt[pt_mask]
        orient_pt = orient_pt[pt_mask]
        orient_vector_pt = orient_vector_pt[pt_mask]
        edge_pt2intersection = radius(pos_pt[pt_intersection_mask], pos_pt[pt_not_intersection_mask], 
               r=2.5, batch_x=data['map_point']['batch'][pt_mask][pt_intersection_mask] if isinstance(data, Batch) else None, 
               batch_y=data['map_point']['batch'][pt_mask][pt_not_intersection_mask] if isinstance(data, Batch) else None)
        
        edge_index_pt2pt = torch.cat([edge_index_pt2pt, edge_pt2intersection], dim=-1)
        if self.dataset == 'argoverse_v2':
            if self.input_dim == 2:
                
                pt_batch = data['map_point']['batch'][pt_mask]
                pt_map_min = data['map_min'].view(-1, 3)[pt_batch, :2]
                pt_map_max = data['map_max'].view(-1, 3)[pt_batch, :2]
                
                scaled_positions_pt = 2* (pos_pt - pt_map_min) / (pt_map_max - pt_map_min) - 1

                pt_magnitude = data['map_point']['magnitude'].unsqueeze(-1)[pt_mask]
                
                x_pt = torch.concat([scaled_positions_pt, orient_pt.unsqueeze(-1), pt_magnitude, curvatures[pt_mask].unsqueeze(-1) ], dim=-1)
                
                x_pl = None
            elif self.input_dim == 3:
                x_pt = torch.stack([data['map_point']['magnitude'], data['map_point']['height']], dim=-1)
                x_pl = data['map_polygon']['height'].unsqueeze(-1)
            else:
                raise ValueError('{} is not a valid dimension'.format(self.input_dim))
            x_pt_categorical_embs = [self.type_pt_emb(data['map_point']['type'][pt_mask].long())]
            x_pl_categorical_embs = [self.type_pl_emb(data['map_polygon']['type'].long()),
                                     self.int_pl_emb(data['map_polygon']['is_intersection'].long())]
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        
        # edge_index_pt2pl[0]: point index, edge_index_pt2pl[1]: polygon index
        
        x_pl = self.x_pl_emb(continuous_inputs=x_pl, categorical_embs=x_pl_categorical_embs)
        x_pt_categorical_embs.append(x_pl[edge_index_pt2pl[1]])
        x_pt = self.x_pt_emb(continuous_inputs=x_pt, categorical_embs=x_pt_categorical_embs)
        
        
        rel_pos_pt2pt = pos_pt[edge_index_pt2pt[0]] - pos_pt[edge_index_pt2pt[1]]
        rel_orient_pt2pt = wrap_angle(orient_pt[edge_index_pt2pt[0]] - orient_pt[edge_index_pt2pt[1]])
        
        if self.input_dim == 2:
            r_pt2pt = torch.stack(
                [torch.norm(rel_pos_pt2pt[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orient_vector_pt[edge_index_pt2pt[1]],
                                          nbr_vector=rel_pos_pt2pt[:, :2]),
                 rel_orient_pt2pt], dim=-1)
        elif self.input_dim == 3:
            r_pt2pt = torch.stack(
                [torch.norm(rel_pos_pt2pt[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orient_vector_pt[edge_index_pt2pt[1]],
                                          nbr_vector=rel_pos_pt2pt[:, :2]),
                 rel_pos_pt2pt[:, -1],
                 rel_orient_pt2pt], dim=-1)
        else:
            raise ValueError('{} is not a valid dimension'.format(self.input_dim))

        r_pt2pt = self.r_pt2pt_emb(continuous_inputs=r_pt2pt, categorical_embs=None)
        
        x_pt_sum = x_pt
        for i in range(self.num_layers):
            x_pt = self.pt2pt_layers[i]((x_pt, x_pt), r_pt2pt, edge_index_pt2pt)
            x_pt_sum = x_pt_sum + x_pt
            
        x_pt = self.out(x_pt_sum)
        return {'x_pt': x_pt, 'x_pl': x_pl}
    
