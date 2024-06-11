from tqdm import tqdm
#import copy
from argparse import ArgumentParser

import json
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from datasets import ArgoverseV2Dataset, ITRIVisualizeWrapper3D
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

def get_agent_features():
    _agent_types = {
        'car': 0,
        'pedestrian': 1,
        'bimo': 2,
        'truck': 4,
        'bus': 4,
    }
    with open('./raw_data/2020-09-11-17-31-33_6/tracking/2020-09-11-17-31-33_6_ImmResult.json', 'r') as f:
        json_data = json.load(f)
    
    num_steps = len(json_data['frames'])  # Assuming 'frames' contains the trajectory data
    unique_tracking_ids = set()
    for ts in json_data['frames']:
        frame_data = json_data['frames'][ts]
        for obj_data in frame_data['objects']:
            unique_tracking_ids.add(obj_data['tracking_id'])

    num_agents = len(unique_tracking_ids) + 1  # Including the AV 

    agent_ids = [None] * num_agents
    agent_type = torch.zeros(num_agents, dtype=torch.uint8)
    agent_category = torch.zeros(num_agents, dtype=torch.uint8)
    position = torch.zeros(num_agents, num_steps, 2, dtype=torch.float)
    heading = torch.zeros(num_agents, num_steps, dtype=torch.float)
    velocity = torch.zeros(num_agents, num_steps, 2, dtype=torch.float)
    valid_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool)
    current_valid_mask = torch.zeros(num_agents, dtype=torch.bool)
    predict_mask = torch.ones(num_agents, num_steps, dtype=torch.bool)

    # Assuming the AV is the first object in the first frame
    av_idx = 0
    agent_ids[0] = 'AV'
    agent_type[0] = 0  # Assuming AV is of type 'car'

    for frame_idx, ts in enumerate(json_data['frames']):
        frame_data = json_data['frames'][ts]
        for obj_idx, obj_data in enumerate(frame_data['objects']):
            agent_idx = obj_idx + 1  # Start from index 1, as index 0 is for the AV
            agent_ids[agent_idx] = obj_data['tracking_id']
            agent_type[agent_idx] = _agent_types[obj_data['tracking_name']]
            # Assuming the category is not provided in the JSON, setting it to 0 (background)
            agent_category[agent_idx] = 0
            position[agent_idx, frame_idx] = torch.tensor([obj_data['translation']['x'], obj_data['translation']['y']])
            rotation_y = torch.tensor(obj_data['rotation']['y'], dtype=torch.float)
            rotation_w = torch.tensor(obj_data['rotation']['w'], dtype=torch.float)
            heading[agent_idx, frame_idx] = torch.atan2(rotation_y, rotation_w) * 2
            velocity[agent_idx, frame_idx] = torch.tensor([obj_data['velocity']['x'], obj_data['velocity']['y']])
            valid_mask[agent_idx, frame_idx] = True

    # Assuming predict mask should be False for all frames for AV
    predict_mask[0, :] = False 
    
    return {
        'num_nodes': num_agents,
        'av_index': av_idx,
        'valid_mask': valid_mask,
        'predict_mask': predict_mask,
        'id': agent_ids,
        'type': agent_type,
        'category': agent_category,
        'position': position,
        'heading': heading,
        'velocity': velocity,
    }

def preprocess_lane_data(waypoints_file):
    with open(waypoints_file, 'r') as f:
        waypoints_raw_data = json.load(f)
    waypoints = waypoints_raw_data['waypoints']
    
    lane_centerlines = []
    for wp in waypoints:
        points = wp['points']
        lane_array = np.array([[point['x'], point['y'], point['z']] for point in points])
        if len(lane_array) > 10:
            split_arrays = split_lane_array(lane_array)
            for array in split_arrays:
                if len(array) == 10:
                    lane_centerlines.append(array)
        elif len(lane_array) == 10:
            lane_centerlines.append(lane_array)
    lane_centerlines = np.array(lane_centerlines)
    
    lane_polygons = []
    split_left_boundary_points = []
    split_right_boundary_points = []
    for waypoint in waypoints:
        points = waypoint['points']
        left_boundary_points = []
        right_boundary_points = []
        for point in points:
            x, y, z, width, angle = point['x'], point['y'], point['z'], point['width'], point['angle']
            half_width = width / 2.0
           
            # Calculate the offsets for the left and right boundaries
            dx = half_width * np.sin(angle)
            dy = half_width * np.cos(angle)
            
            left_x = x + dx
            left_y = y - dy
            right_x = x - dx
            right_y = y + dy
            
            left_boundary_points.append([left_x, left_y, z])
            right_boundary_points.append([right_x, right_y, z])
        assert len(left_boundary_points) == len(right_boundary_points)
        
        if len(left_boundary_points) > 10:
            split_left_arrays = split_lane_array(left_boundary_points)
            split_right_arrays = split_lane_array(right_boundary_points)
            for array in split_left_arrays:
                if len(array) == 10:
                    split_left_boundary_points.append(array)
            for array in split_right_arrays:
                if len(array) == 10:
                    split_right_boundary_points.append(array)
        elif len(left_boundary_points) == 10:
            split_left_boundary_points.append(left_boundary_points)
            split_right_boundary_points.append(right_boundary_points)

        #for lp, rp in zip(split_left_boundary_points, split_right_boundary_points):
        #    lane_polygons.append(np.array(lp + rp[::-1] + [lp[0]]))
    #lane_polygons = np.array(lane_polygons)
    left_boundary = np.array(split_left_boundary_points)
    right_boundary = np.array(split_right_boundary_points)
    
    return lane_centerlines, left_boundary, right_boundary 

def split_lane_array(lane_array, chunk_size=10):
    chunks = [lane_array[i:i + chunk_size] for i in range(0, len(lane_array), chunk_size)]
    return chunks

def get_map_features():
    waypoints_file = './raw_data/hd_map/waypoints.json'
    lane_centerlines, left_boundaries, right_boundaries = preprocess_lane_data(waypoints_file)
    assert lane_centerlines.shape[0] == left_boundaries.shape[0]
    assert right_boundaries.shape[0] == left_boundaries.shape[0]

    _polygon_types = ['VEHICLE', 'BIKE', 'BUS', 'PEDESTRIAN']
    _polygon_is_intersections = [True, False, None]
    _point_types = ['DASH_SOLID_YELLOW', 'DASH_SOLID_WHITE', 'DASHED_WHITE', 'DASHED_YELLOW',
                    'DOUBLE_SOLID_YELLOW', 'DOUBLE_SOLID_WHITE', 'DOUBLE_DASH_YELLOW', 'DOUBLE_DASH_WHITE',
                    'SOLID_YELLOW', 'SOLID_WHITE', 'SOLID_DASH_WHITE', 'SOLID_DASH_YELLOW', 'SOLID_BLUE',
                    'NONE', 'UNKNOWN', 'CROSSWALK', 'CENTERLINE']
    _point_sides = ['LEFT', 'RIGHT', 'CENTER']
    _polygon_to_polygon_types = ['NONE', 'PRED', 'SUCC', 'LEFT', 'RIGHT']

    dim = 3
    lane_segment_ids = [i for i in range(len(lane_centerlines))]
    polygon_ids = lane_segment_ids
    num_polygons = len(lane_segment_ids)

    # initialization
    polygon_position = torch.zeros(num_polygons, dim, dtype=torch.float)
    polygon_orientation = torch.zeros(num_polygons, dtype=torch.float)
    polygon_height = torch.zeros(num_polygons, dtype=torch.float)
    polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
    polygon_is_intersection = torch.zeros(num_polygons, dtype=torch.uint8)
    point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_orientation: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_magnitude: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_height: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_type: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_side: List[Optional[torch.Tensor]] = [None] * num_polygons

    for lane_segment_idx, lane_segment in enumerate(lane_centerlines):
        centerline = torch.from_numpy(lane_centerlines[lane_segment_idx]).float()
        polygon_position[lane_segment_idx] = centerline[0, :dim]
        polygon_orientation[lane_segment_idx] = torch.atan2(centerline[1, 1] - centerline[0, 1],
                                                            centerline[1, 0] - centerline[0, 0])
        polygon_height[lane_segment_idx] = centerline[1, 2] - centerline[0, 2]
        polygon_type[lane_segment_idx] = _polygon_types.index('VEHICLE')
        polygon_is_intersection[lane_segment_idx] = _polygon_is_intersections.index(None)
        
        left_boundary = torch.from_numpy(left_boundaries[lane_segment_idx][:, :dim]).float()
        right_boundary = torch.from_numpy(right_boundaries[lane_segment_idx][:, :dim]).float()
        point_position[lane_segment_idx] = torch.cat([left_boundary[:-1],
                                                      right_boundary[:-1],
                                                      centerline[:-1]], dim=0)
        left_vectors = left_boundary[1:] - left_boundary[:-1]
        right_vectors = right_boundary[1:] - right_boundary[:-1]
        center_vectors = centerline[1:] - centerline[:-1]
        point_orientation[lane_segment_idx] = torch.cat([torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                                                         torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                                                         torch.atan2(center_vectors[:, 1], center_vectors[:, 0])],
                                                        dim=0)
        point_magnitude[lane_segment_idx] = torch.norm(torch.cat([left_vectors,
                                                                  right_vectors,
                                                                  center_vectors], dim=0), p=2, dim=-1)
        point_height[lane_segment_idx] = torch.cat([left_boundary[:, 2],
                                                    right_boundary[:, 2],
                                                    centerline[:, 2]], dim=0)
        left_type = _point_types.index('DASHED_WHITE')
        right_type = _point_types.index('DASHED_WHITE')
        center_type = _point_types.index('CENTERLINE')
        point_type[lane_segment_idx] = torch.cat(
            [torch.full((len(left_vectors),), left_type, dtype=torch.uint8),
             torch.full((len(right_vectors),), right_type, dtype=torch.uint8),
             torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
        point_side[lane_segment_idx] = torch.cat(
            [torch.full((len(left_vectors),), _point_sides.index('LEFT'), dtype=torch.uint8),
             torch.full((len(right_vectors),), _point_sides.index('RIGHT'), dtype=torch.uint8),
             torch.full((len(center_vectors),), _point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)

    num_points = torch.tensor([point.size(0) for point in point_position], dtype=torch.long)
    point_to_polygon_edge_index = torch.stack(
        [torch.arange(num_points.sum(), dtype=torch.long),
         torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points)], dim=0)
    
    polygon_to_polygon_edge_index = torch.tensor([[], []], dtype=torch.long)
    polygon_to_polygon_type = torch.tensor([], dtype=torch.uint8)

    map_data = {
            'map_polygon': {},
            'map_point': {},
            ('map_point', 'to', 'map_polygon'): {},
            ('map_polygon', 'to', 'map_polygon'): {},
        }
    map_data['map_polygon']['num_nodes'] = num_polygons
    map_data['map_polygon']['position'] = polygon_position
    map_data['map_polygon']['orientation'] = polygon_orientation
    if dim == 3:
        map_data['map_polygon']['height'] = polygon_height
    map_data['map_polygon']['type'] = polygon_type
    map_data['map_polygon']['is_intersection'] = polygon_is_intersection
    if len(num_points) == 0:
        map_data['map_point']['num_nodes'] = 0
        map_data['map_point']['position'] = torch.tensor([], dtype=torch.float)
        map_data['map_point']['orientation'] = torch.tensor([], dtype=torch.float)
        map_data['map_point']['magnitude'] = torch.tensor([], dtype=torch.float)
        if dim == 3:
            map_data['map_point']['height'] = torch.tensor([], dtype=torch.float)
        map_data['map_point']['type'] = torch.tensor([], dtype=torch.uint8)
        map_data['map_point']['side'] = torch.tensor([], dtype=torch.uint8)
    else:
        map_data['map_point']['num_nodes'] = num_points.sum().item()
        map_data['map_point']['position'] = torch.cat(point_position, dim=0)
        map_data['map_point']['orientation'] = torch.cat(point_orientation, dim=0)
        map_data['map_point']['magnitude'] = torch.cat(point_magnitude, dim=0)
        if dim == 3:
            map_data['map_point']['height'] = torch.cat(point_height, dim=0)
        map_data['map_point']['type'] = torch.cat(point_type, dim=0)
        map_data['map_point']['side'] = torch.cat(point_side, dim=0)
    map_data['map_point', 'to', 'map_polygon']['edge_index'] = point_to_polygon_edge_index
    map_data['map_polygon', 'to', 'map_polygon']['edge_index'] = polygon_to_polygon_edge_index
    map_data['map_polygon', 'to', 'map_polygon']['type'] = polygon_to_polygon_type
    return map_data

def preprocess_data():
    data = dict()
    data['scenario_id'] = '0001'
    data['city'] = 'Hsinchu'
    data['agent'] = get_agent_features()
    data.update(get_map_features()) 
    return data

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
    
    preprocess_data = preprocess_data()
    viz_wrapper = ITRIVisualizeWrapper3D()
    viz_wrapper.forward()
    exit()

    model = {
        'TempEnsLearnAgg': TempEnsLearnAgg,
    }[args.model].load_from_checkpoint(checkpoint_path=args.ckpt_path, map_location='cpu')
    val_dataset = {
        'argoverse_v2': ArgoverseV2Dataset,
    }[model.dataset](root=args.root, split='val',
                     transform=TargetBuilder(model.num_historical_steps, model.num_future_steps))
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    
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
