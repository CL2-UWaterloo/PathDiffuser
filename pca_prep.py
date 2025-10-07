import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
import numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,4'

from argparse import ArgumentParser

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import ArgoverseV2Dataset
from predictors import QCNet, DiffNet
from transforms import TargetBuilder
import torch
from tqdm import tqdm

if __name__ == '__main__':
    pl.seed_everything(2024, workers=True)
    
    
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16) 
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--devices', type=str, default="1")
    
    args = parser.parse_args()
    # model = DiffNet.load_from_checkpoint(checkpoint_path=args.ckpt_path)

    # model.add_extra_param(args)
    num_historical_steps = 50
    num_future_steps = 60
    output_dim = 2
    # mode_list = ['train', 'val', 'test']
    mode_list = ['train']
    for mode in mode_list:
        # mode =  'test'
        # data_accumulate = torch.zeros((0, 2))

        data_accumulate = []
        dataset = ArgoverseV2Dataset(root=args.root, split=mode,
                        transform=TargetBuilder(num_historical_steps, num_future_steps))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)

        for batch in tqdm(loader, desc="Processing Batches"):
            # find non-vehicle agents, and out of scene agents
            reg_mask = batch['agent']['predict_mask'][:, num_historical_steps:]
            mask = (batch['agent']['category'] >= 2) & (reg_mask[:,-1] == True) & (reg_mask[:,0] == True)
            # if data_accumulate.device != mask.device:
            #     data_accumulate = data_accumulate.to(mask.device)
            # get GT for the target agent and mask out agents
            # gt = torch.cat([batch['agent']['translation_only'][..., :output_dim], batch['agent']['target'][..., -1:]], dim=-1)            
            gt_delta_trans = batch['agent']['delta_translation'][mask, :output_dim]   
            gt_delta_rot = batch['agent']['delta_angle'][mask].unsqueeze(-1)   
            # gt = gt[mask][..., :output_dim]
            gt_delta = torch.cat([gt_delta_trans, gt_delta_rot], dim=-1)
            # flat_gt = gt.reshape(gt.size(0), -1)
            data_accumulate.append(gt_delta.detach())
            # data_accumulate = torch.cat([data_accumulate, gt_delta_trans], dim=0)
            # print(data_accumulate.mean(dim=0), data_accumulate.std(dim=0))
        data_accumulate = torch.cat(data_accumulate, dim=0)
        mean_std = torch.stack([data_accumulate.mean(dim=0), data_accumulate.std(dim=0)])
        print(data_accumulate.mean(dim=0), data_accumulate.std(dim=0))
        np.save(f'./hq_gt_score_1_delta_mean_std_{mode}.npy', mean_std.cpu().numpy())
        
        # data_accumulate = torch.cat(data_accumulate, dim=0).cpu().numpy()
        # np.save(f'./ours_pca/hq_gt_score_1_delta_{mode}.npy', data_accumulate)