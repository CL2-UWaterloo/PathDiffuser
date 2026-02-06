# Path Diffuser: Diffusion Model for Data-Driven Traffic Simulator
This repository contains the implementation of [Path Diffuser](https://dasaemlee.github.io/projects/pathdiffuser/) ([arXiv](https://arxiv.org/abs/2509.24995)), which was accepted to ITSC 2025.

  
## Abstract
> Simulating diverse and realistic traffic scenarios is critical for developing and testing autonomous planning. Traditional rule-based planners lack diversity and realism, while learning-based simulators often replay, forecast, or edit scenarios using historical agent trajectories. However, they struggle to generate new scenarios, limiting scalability and diversity due to their reliance on fully annotated logs and historical data. Thus, a key challenge for a learning-based simulator's performance is that it requires agents' past trajectories and pose information in addition to map data, which might not be available for all agents on the road.Without which, generated scenarios often produce unrealistic trajectories that deviate from drivable areas, particularly under out-of-distribution (OOD) map scenes (e.g., curved roads). To address this, we propose Path Diffuser (PD): a two-stage, diffusion model for generating agent pose initializations and their corresponding trajectories conditioned on the map, free of any historical context of agents' trajectories. Furthermore, PD incorporates a motion primitive-based prior, leveraging Frenet frame candidate trajectories to enhance diversity while ensuring road-compliant trajectory generation. We also explore various design choices for modeling complex multi-agent interactions. We demonstrate the effectiveness of our method through extensive experiments on the Argoverse2 Dataset and additionally evaluate the generalizability of the approach on OOD map variants. Notably, Path Diffuser outperforms the baseline methods by 1.92x on distribution metrics, 1.14x on common-sense metrics, and 1.62x on road compliance from adversarial benchmarks.
</p>

## Install

**Step 1**: Download the code by cloning the repository:
```
git clone https://github.com/CL2-UWaterloo/PathDiffuser && cd PathDiffuser
```

**Step 2**: Set up a new conda environment and install required packages:
```
conda env create -f environment.yml
conda activate PathDiffuser
```

**Step 3**: Implement the [Argoverse 2 API](https://github.com/argoverse/av2-api) and access the [Argoverse 2 Motion Forecasting Dataset](https://www.argoverse.org/av2.html). Please see the [Argoverse 2 User Guide](https://argoverse.github.io/user-guide/getting_started.html).



## Agent Initialization

### Training Command
```sh
python train_diffnet_tb.py --root <Path to dataset> --train_batch_size 32 --val_batch_size 32 --test_batch_size 32 --dataset argoverse_v2 --init_timestep 50 --pl2pl_radius 150 --devices 1 --num_workers 16 --num_denoiser_layers 3 --num_diffusion_steps 100 --max_epochs 30 --lr 0.001 --beta_1 0.0001 --beta_T 0.05 --diff_type vd --sampling ddpm --sampling_stride 10 --num_eval_samples 1 --check_val_every_n_epoch 1 --stage init
```
Below are the significant arguments related to our work:

- `--devices`: Specifies the GPUs you want to use.
- `--num_denoiser_layers`: Defines the number of layers in the diffusion network.
- `--num_diffusion_steps`: Sets the number of diffusion steps.
- `--max_epochs`: Determines the total number of training epochs.
- `--lr`: Sets the learning rate.
- `--beta_1`: Specifies the  $\beta_1$, the diffusion schedule parameter.
- `--beta_T`: Specifies the $\beta_T$, the diffusion schedule parameter.
- `--sampling_stride`: Defines the sampling stride for DDIM.
- `--num_eval_samples`: Indicates the number of evaluation samples.
- `--init_timestep`: Specifies the initial timestep to use for training.



### Validation Command
```sh
python val_diffnet.py --root <Path to dataset> --ckpt_path <Path to diffusion network checkpoint> --devices '1' --batch_size 8 --sampling ddim --sampling_stride 10 --num_eval_samples 128 --network_mode 'val' --stage init
```

## Trajectory Generation

### Training Command
```sh
python train_diffnet_tb.py --root  <Path to dataset> --train_batch_size 32 --val_batch_size 32 --test_batch_size 32 --dataset argoverse_v2 --init_timestep 50 --num_generation_timestep 60 --pl2pl_radius 150 --devices 1 --qcnet_map_ckpt_path  <Path to diffusion network checkpoint>  --num_workers 16 --num_denoiser_layers 3 --num_diffusion_steps 100 --max_epochs 30 --lr 0.005 --beta_1 0.0001 --beta_T 0.05 --diff_type vd --sampling ddpm --sampling_stride 10 --num_eval_samples 1 --check_val_every_n_epoch 1 --path_pca_s_mean pca/imp_org/s_mean_10.npy --path_pca_VT_k pca/imp_org/VT_k_10.npy --path_pca_V_k pca/imp_org/V_k_10.npy --path_pca_latent_mean pca/imp_org/latent_mean_10.npy --path_pca_latent_std pca/imp_org/latent_std_10.npy --stage traj
```
Below are the significant arguments related to our work:

- `--devices`: Specifies the GPUs you want to use.
- `--qcnet_map_ckpt_path`: Provides the path to the QCNet checkpoints.
- `--num_denoiser_layers`: Defines the number of layers in the diffusion network.
- `--num_diffusion_steps`: Sets the number of diffusion steps.
- `--max_epochs`: Determines the total number of training epochs.
- `--lr`: Sets the learning rate.
- `--beta_1`: Specifies the  $\beta_1$, the diffusion schedule parameter.
- `--beta_T`: Specifies the $\beta_T$, the diffusion schedule parameter.
- `--sampling_stride`: Defines the sampling stride for DDIM.
- `--num_eval_samples`: Indicates the number of evaluation samples.
- `--init_timestep`: Specifies the initial timestep to use for training.
- `--num_generation_timestep`: Specifies the number of timestep to generate.


### Validation Command
```sh
python val_diffnet.py --root <Path to dataset> --ckpt_path <Path to diffusion network checkpoint> --devices '1' --batch_size 8 --sampling ddim --sampling_stride 10 --num_eval_samples 128 --path_pca_V_k 'pca/imp_org/V_k_10.npy' --network_mode 'val' --stage traj
```

## Citation
```
@misc{lee2025pathdiffuserdiffusionmodel,
      title={Path Diffuser: Diffusion Model for Data-Driven Traffic Simulator}, 
      author={Da Saem Lee and Akash Karthikeyan and Yash Vardhan Pant and Sebastian Fischmeister},
      year={2025},
      eprint={2509.24995},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.24995}, 
}
```

## Acknowledgement

This code is based on [Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation](https://github.com/YixiaoWang7/OptTrajDiff) and [Query-Centric Trajectory Prediction](https://github.com/ZikangZhou/QCNet).
Please also consider citing:

```
@article{wang2024optimizing,
  title={Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation},
  author={Wang, Yixiao and Tang, Chen and Sun, Lingfeng and Rossi, Simone and Xie, Yichen and Peng, Chensheng and Hannagan, Thomas and Sabatini, Stefano and Poerio, Nicola and Tomizuka, Masayoshi and others},
  journal={arXiv preprint arXiv:2408.00766},
  year={2024}
}
@inproceedings{zhou2023query,
  title={Query-Centric Trajectory Prediction},
  author={Zhou, Zikang and Wang, Jianping and Li, Yung-Hui and Huang, Yu-Kai},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```
