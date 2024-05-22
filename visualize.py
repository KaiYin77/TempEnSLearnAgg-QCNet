from tqdm import tqdm
#import copy
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from datasets import ArgoverseV2Dataset, ArgoverseV2VisualizeWrapper3D
from predictors import TempEnsLearnAgg 
from transforms import TargetBuilder

from av2.datasets.motion_forecasting.eval.metrics import compute_ade, compute_fde

def postprocess(data, traj_refine, pi, time_shift=10):
    num_historical_steps = 50
    eval_mask = data['agent']['category'] == 3

    origin_eval = data['agent']['position'][eval_mask, num_historical_steps - 1 + time_shift]
    theta_eval = data['agent']['heading'][eval_mask, num_historical_steps - 1 + time_shift]
    cos, sin = theta_eval.cos(), theta_eval.sin()
    rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device='cpu')
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = sin
    rot_mat[:, 1, 0] = -sin
    rot_mat[:, 1, 1] = cos
    traj_eval = torch.matmul(traj_refine[eval_mask, :, :, :2],
                             rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)

    # slicing traj_eval into global time_span [10, 60) evaluate at t=10
    batch_sliced_traj_eval = traj_eval[:, :, 10-time_shift: 10-time_shift+50]
    batch_pi_eval = pi[eval_mask]
    return batch_sliced_traj_eval, batch_pi_eval

    
def predict(data, model, time_shift=10):
    if isinstance(data, Batch):
        data['agent']['av_index'] += data['agent']['ptr'][:-1]
    reg_mask = data['agent']['predict_mask'][:, 50:]
    cls_mask = data['agent']['predict_mask'][:, -1]
    pred, scene_enc = model(data, time_shift)

    output_head = True
    output_dim = 2
    if output_head:
        traj_propose = torch.cat([pred['loc_propose_pos'][..., :output_dim],
                                  pred['loc_propose_head'],
                                  pred['scale_propose_pos'][..., :output_dim],
                                  pred['conc_propose_head']], dim=-1)
        traj_refine = torch.cat([pred['loc_refine_pos'][..., :output_dim],
                                 pred['loc_refine_head'],
                                 pred['scale_refine_pos'][..., :output_dim],
                                 pred['conc_refine_head']], dim=-1)
    else:
        traj_propose = torch.cat([pred['loc_propose_pos'][..., :output_dim],
                                  pred['scale_propose_pos'][..., :output_dim]], dim=-1)
        traj_refine = torch.cat([pred['loc_refine_pos'][..., :output_dim],
                                 pred['scale_refine_pos'][..., :output_dim]], dim=-1)
    pi = pred['pi']
    gt = torch.cat([data['agent']['target'][..., :output_dim], data['agent']['target'][..., -1:]], dim=-1)

    return traj_refine[..., :output_dim], pi, pred['m'], scene_enc

def predict_learning_based_aggregation(model, m, data, scene_enc, time_shift=10):

    pred = model.temporal_aggregate_layer(m, data, scene_enc, time_shift)
    
    output_head = True
    output_dim = 2
    if output_head:
        traj_propose = torch.cat([pred['loc_propose_pos'][..., :output_dim],
                                  pred['loc_propose_head'],
                                  pred['scale_propose_pos'][..., :output_dim],
                                  pred['conc_propose_head']], dim=-1)
        traj_refine = torch.cat([pred['loc_refine_pos'][..., :output_dim],
                                 pred['loc_refine_head'],
                                 pred['scale_refine_pos'][..., :output_dim],
                                 pred['conc_refine_head']], dim=-1)
    else:
        traj_propose = torch.cat([pred['loc_propose_pos'][..., :output_dim],
                                  pred['scale_propose_pos'][..., :output_dim]], dim=-1)
        traj_refine = torch.cat([pred['loc_refine_pos'][..., :output_dim],
                                 pred['scale_refine_pos'][..., :output_dim]], dim=-1)
    pi = pred['pi']

    return traj_refine[..., :output_dim], pi

def calculate_metrics(temp_sliced, gt_sliced):
    fde_k = compute_fde(temp_sliced, gt_sliced)
    ade_k = compute_ade(temp_sliced, gt_sliced)
    min_fde = fde_k.min()
    min_ade = ade_k.min()
    miss_rate = 1 if min_fde > 2.0 else 0
    return min_fde, min_ade, miss_rate

if __name__ == '__main__':
    pl.seed_everything(2024, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1) # please fix to batch_size=1
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()

    model = {
        'TempEnsLearnAgg': TempEnsLearnAgg,
    }[args.model].load_from_checkpoint(checkpoint_path=args.ckpt_path, map_location='cpu')
    val_dataset = {
        'argoverse_v2': ArgoverseV2Dataset,
    }[model.dataset](root=args.root, split='val',
                     transform=TargetBuilder(model.num_historical_steps, model.num_future_steps))
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    viz_wrapper = ArgoverseV2VisualizeWrapper3D()
    
    data_iter = tqdm(dataloader)
    for data in data_iter:
        # 1. Temporal Ensembling: Gathering Historical Prediction of Mode Queries
        batch_sliding_window_trajs = []
        batch_sliding_window_clsfs = []
        batch_sliding_window_m = []
        for time_shift in range(1, 11): # Gather Most Recent 10 Frames
            # Model Output Coordinate: Current-Agent-Centric
            traj_refine, pi, m, scene_enc = predict(data, model, time_shift)
            # Post-process: Tranform them from Current-Agent-Centric to World coordinate
            batch_sliced_traj_eval, batch_pi_eval = postprocess(data, traj_refine, pi, time_shift=time_shift)
            
            batch_sliding_window_trajs.append(batch_sliced_traj_eval)
            batch_sliding_window_clsfs.append(batch_pi_eval)
            batch_sliding_window_m.append(m)
        batch_sliding_window_trajs = torch.stack(batch_sliding_window_trajs, dim=1)
        batch_sliding_window_clsfs = torch.stack(batch_sliding_window_clsfs, dim=1)
        batch_sliding_window_m = torch.stack(batch_sliding_window_m, dim=1)

        # 2. Learn-based Aggregation
        # Model Output Coordinate: Current-Agent-Centric
        batch_tela_traj, batch_tela_pi = predict_learning_based_aggregation(model, batch_sliding_window_m, data, scene_enc, time_shift=10)
        # Post-process: Tranform them from Current-Agent-Centric to World coordinate
        batch_tela_traj_eval, batch_tela_pi_eval = postprocess(data, batch_tela_traj, batch_tela_pi, time_shift=10)

        # 3. Visualize Target Agent Prediction
        eval_mask = data['agent']['category'] == 3 # _agent_categories==3 means focal_track
        batch_gt_eval = data['agent']['position'][eval_mask, 60:, :2] # slice [60, 110) agent position to be groundtruth
        for idx in range(batch_gt_eval.shape[0]):
            single_frame_pred_trajs_eval = batch_sliding_window_trajs[idx][-1].reshape(-1, 50, 2).detach().cpu().numpy().copy()
            sliding_window_pred_trajs_eval = batch_sliding_window_trajs[idx][:].reshape(-1, 50, 2).detach().cpu().numpy().copy()
            tela_pred_trajs_eval = batch_tela_traj_eval[idx].reshape(-1, 50, 2).detach().cpu().numpy()
            gt_traj_eval = batch_gt_eval[idx].detach().cpu().numpy()
            viz_wrapper.forward({
                'processed_data': data,
                'baseline_sf_pred_trajs': single_frame_pred_trajs_eval,
                'baseline_sw_pred_trajs': sliding_window_pred_trajs_eval,
                'tela_pred_trajs': tela_pred_trajs_eval,
                'gt_traj': gt_traj_eval,
                'dataset': val_dataset
            })
