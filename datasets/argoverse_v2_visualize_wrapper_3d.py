import os
from pathlib import Path

import torch
import numpy as np
import pandas as pd 
import torch
from scipy.spatial.distance import cdist

import matplotlib
from matplotlib import colors, cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as patches
from matplotlib.patches import Circle, Ellipse
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from av2.datasets.motion_forecasting import scenario_serialization
from av2.geometry.interpolate import compute_midpoint_line
import av2.geometry.polyline_utils as polyline_utils
from av2.map.map_api import ArgoverseStaticMap
from av2.map.map_primitives import Polyline
from av2.utils.io import read_json_file

from av2.datasets.motion_forecasting.eval.metrics import compute_ade, compute_fde

# for saving to png
plt.switch_backend('agg')

_SIZE = {
    "vehicle": [4.0, 2.0, 1.8],
    "pedestrian": [0.7, 0.7, 1.7],
    "motorcyclist": [2.0, 0.7, 1.6],
    "cyclist": [2.0, 0.7, 1.6],
    "bus": [7.0, 3.0, 3.8],
    "static": [0.0, 0.0, 0.0],
    "background": [0.0, 0.0, 0.0],
    "construction": [0.0, 0.0, 0.0],
    "riderless_bicycle": [2.0, 0.7, 0.6],
    "unknown": [0.0, 0.0, 0.0],
}

def calculate_metrics(temp_sliced, gt_sliced):
    fde_k = compute_fde(temp_sliced, gt_sliced)
    ade_k = compute_ade(temp_sliced, gt_sliced)
    min_fde = fde_k.min()
    min_ade = ade_k.min()
    miss_rate = 1 if min_fde > 2.0 else 0
    return min_fde, min_ade, miss_rate

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D, 'arrow3D', _arrow3D)

class ArgoverseV2VisualizeWrapper3D:
    def __init__(self):
        self.map_api = ArgoverseStaticMap

    def get_map_features(self, map_api, centerlines):
        lane_segment_ids = map_api.get_scenario_lane_segment_ids()
        cross_walk_ids = list(map_api.vector_pedestrian_crossings.keys())
        polygon_ids = lane_segment_ids + cross_walk_ids
        
        lane_segments = map_api.get_scenario_lane_segments()
        lane_centerlines = np.array([list(map_api.get_lane_segment_centerline(s.id)) for s in lane_segments])
        lane_polygons = np.array([polyline_utils.centerline_to_polygon(c, visualize=False) for c in lane_centerlines])
        
        crosswalk_segments = map_api.get_scenario_ped_crossings()
        crosswalk_polygons = np.array([s.polygon[:, :2] for s in crosswalk_segments])

        return {
            'lane_centerlines': lane_centerlines, 
            'lane_polygons': lane_polygons, 
            'crosswalk_polygons': crosswalk_polygons 
        }
    
    def preprocess_map(self, dataset, processed_data):
        raw_file_name = processed_data['scenario_id'][0]
        df = pd.read_parquet(os.path.join(dataset.raw_dir, raw_file_name, f'scenario_{raw_file_name}.parquet'))
        map_dir = Path(dataset.raw_dir) / raw_file_name
        map_path = map_dir / sorted(map_dir.glob('log_map_archive_*.json'))[0]
        map_data = read_json_file(map_path)
        centerlines = {lane_segment['id']: Polyline.from_json_data(lane_segment['centerline'])
                        for lane_segment in map_data['lane_segments'].values()}
        map_api = self.map_api.from_json(map_path)
        map_data = self.get_map_features(map_api, centerlines)
        
        return map_data
    
    def matplot_map(self, ax, map_data):
        lane_centerlines = map_data['lane_centerlines']
        for i, l_c in enumerate(lane_centerlines):
            ax.plot(
                l_c[:, 0], l_c[:, 1], 
                ':',
                zs=0, zdir='z',
                color='#0A1931',
                linewidth=0.3,
                alpha=1,
            )
        lane_polygons = map_data['lane_polygons']
        for i, l_p in enumerate(lane_polygons):
            ax.plot(
                l_p[:, 0], l_p[:, 1], zs=0, zdir='z',
                color='dimgray',
                linewidth=0.5,
                label='Map',
                alpha=0.3,
            )
        crosswalk_polygons = map_data['crosswalk_polygons']
        for c_p in crosswalk_polygons:
            ax.plot(
                c_p[:, 0], c_p[:, 1], zs=0, zdir='z',
                color='dimgray',
                linewidth=0.5,
                alpha=0.3,
            )

    def create_fig_and_ax(self, size_pixels, processed_data, gt_traj, sf_trajs, sw_trajs, k_trajs):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        dpi = 100
        fig.set_dpi(dpi)
        fig.set_facecolor('white')
        ax.set_facecolor('white')
        
        fig.set_tight_layout(True)
        ax.grid(False)
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_xticklabels([])
        
        ax.set_yticks([])
        ax.set_yticklabels([])

        sf_fde, sf_ade, sf_miss = calculate_metrics(sf_trajs, gt_traj)
        sw_fde, sw_ade, sw_miss = calculate_metrics(sw_trajs, gt_traj)
        k_fde, k_ade, k_miss = calculate_metrics(k_trajs, gt_traj)
        
        scenario_id = processed_data['scenario_id'][0]
        title = f'scenario_id: {scenario_id}\n'
        title += f'Baseline minADE: {sf_ade.min(): .3f} / minFDE: {sf_fde.min(): .3f}\n'
        title += f'TempEns-LearnAgg minADE: {k_ade.min(): .3f} / minFDE: {k_fde.min(): .3f}\n'
        ax.set_title(title, fontsize=7, color="black")

        canva_center = gt_traj[-1]
        ax.set_xlim(canva_center[0]-20, canva_center[0]+20)
        ax.set_ylim(canva_center[1]-20, canva_center[1]+20)
        ax.set_axis_off()
        ax.grid(False)
        ax.view_init(elev=30, azim=-60)
        
        return fig, ax

    def preprocess_agent(self, dataset, processed_data, time):
        raw_file_name = processed_data['scenario_id'][0]
        scenario_path = os.path.join(dataset.raw_dir, raw_file_name, f'scenario_{raw_file_name}.parquet')
        scenario = scenario_serialization.load_argoverse_scenario_parquet(
            scenario_path)
        tracks_df = scenario_serialization._convert_tracks_to_tabular_format(
            scenario.tracks)
        target_df = tracks_df[tracks_df['object_category'] == 3]
        target_id = target_df['track_id'].to_numpy()[0]
        agent_tracks = []
        for track in scenario.tracks:
            actor_timestep = torch.IntTensor(
                [s.timestep for s in track.object_states])
            observed = torch.Tensor([s.observed for s in track.object_states])

            if (50-1 + time) not in actor_timestep:
                continue
            actor_state = torch.Tensor(
                [list(object_state.position+(object_state.heading, ))
                 for object_state in track.object_states if object_state.timestep == 50-1+time]
            )

            if (track.track_id == target_id):
                target = True
            else:
                target = False
            if not track.object_type in ['vehicle', 'pedestrian', 'motorcyclist', 'bus', 'cyclist']:
                continue
            state = {
                "l": _SIZE[track.object_type][0],
                "w": _SIZE[track.object_type][1],
                "h": _SIZE[track.object_type][2],
                "actor_state": actor_state,
                "object_type": track.object_type,
                "target": target
            }
            agent_tracks.append(state)
        return agent_tracks, raw_file_name

    def matplot_agent(self, ax, agent_data):
        
        for track in agent_data:
            l = track['l']
            w = track['w']
            h = track['h'] / 30
            x = track['actor_state'][0, 0]
            y = track['actor_state'][0, 1]
            theta = track['actor_state'][0, 2] * 180 / np.pi
            target = track['target']

            # 3D Transformation
            ts = ax.transData
            tr = matplotlib.transforms.Affine2D().rotate_deg_around(x, y, theta)
            t = tr + ts

            # Define the 3D rectangle vertices
            vertices = np.array([
                [x - l / 2, y - w / 2, 0],
                [x + l / 2, y - w / 2, 0],
                [x + l / 2, y + w / 2, 0],
                [x - l / 2, y + w / 2, 0],
                [x - l / 2, y - w / 2, h],
                [x + l / 2, y - w / 2, h],
                [x + l / 2, y + w / 2, h],
                [x - l / 2, y + w / 2, h]
            ])

            # Rotate the vertices
            rot_mat = np.radians(theta)
            rotation_matrix = np.array([
                [np.cos(rot_mat), -np.sin(rot_mat), 0],
                [np.sin(rot_mat), np.cos(rot_mat), 0],
                [0, 0, 1]
            ])
            rotated_vertices = np.dot(vertices - [x, y, 0], rotation_matrix.T) + [x, y, 0]

            # Define the faces of the 3D rectangle
            faces = [[rotated_vertices[j] for j in [0, 1, 2, 3]],
                     [rotated_vertices[j] for j in [4, 5, 6, 7]],
                     [rotated_vertices[j] for j in [0, 3, 7, 4]],
                     [rotated_vertices[j] for j in [1, 2, 6, 5]],
                     [rotated_vertices[j] for j in [0, 1, 5, 4]],
                     [rotated_vertices[j] for j in [2, 3, 7, 6]]]

            # Create a Poly3DCollection
            if target: 
                edgecolor = "#A13033" # 'royalblue'
                facecolor = "#ED8199"
            else:
                edgecolor = "#3C736D"
                facecolor = "#94D7D1"

            poly3d = Poly3DCollection(faces, linewidths=0.4, edgecolors=edgecolor, facecolors=facecolor, alpha=0.3)
            ax.add_collection3d(poly3d)
    
    def plot_history(self, ax, time, processed_data):
        past_trajs = processed_data['agent']['position'].detach().cpu().numpy()
        eval_mask = processed_data['agent']['category'] == 3
        past_traj = past_trajs[eval_mask, :50+time, :2][0]
        
        color = 'red'
        ax.plot(
            past_traj[:, 0], past_traj[:, 1],
            color=color,
            linewidth=0.5,
        )

    def plot_bsf_prediction(self, ax, pred_traj):
        pred_traj = pred_traj.reshape(6, 50, 2)
        for i in range(pred_traj.shape[0]):
            color = 'orange'
            ax.plot(
                pred_traj[i, :, 0], pred_traj[i, :, 1], zs=0, zdir='z',
                color=color,
                linestyle='dashed',
                linewidth=0.5,
                alpha=1,
            )
            dx = pred_traj[i, -1, 0] - pred_traj[i, -2, 0]
            dy = pred_traj[i, -1, 1] - pred_traj[i, -2, 1]
            ax.arrow3D(
                pred_traj[i, -1, 0], pred_traj[i, -1, 1], 0,
                dx, dy, 0,
                mutation_scale=5,
                linewidth=0.5,
                fc=color,
                ec=color,
                label='Baseline trajs'
            )

    def plot_bsw_prediction(self, ax, pred_traj):
        pred_traj = pred_traj.reshape(-1, 50, 2)
        for i in range(pred_traj.shape[0]):
            color = 'dimgrey'
            ax.plot(
                pred_traj[i, :, 0], pred_traj[i, :, 1], zs=0.2, zdir='z',
                color=color,
                linestyle='dashed',
                linewidth=0.15,
                alpha=1,
            )
            dx = pred_traj[i, -1, 0] - pred_traj[i, -2, 0]
            dy = pred_traj[i, -1, 1] - pred_traj[i, -2, 1]
            ax.arrow3D(
                pred_traj[i, -1, 0], pred_traj[i, -1, 1], 0.2,
                dx, dy, 0,
                mutation_scale=3,
                linewidth=0.15,
                fc=color,
                ec=color,
                label='K-sweep=10 trajs'
            )

    def plot_tela_prediction(self, ax, pred_traj):
        pred_traj = pred_traj.reshape(6, 50, 2)

        for i in range(pred_traj.shape[0]):
            color = 'royalblue'#'orange'
            ax.plot(
                pred_traj[i, :, 0], pred_traj[i, :, 1], zs=0, zdir='z', 
                color=color,
                linestyle='dashed',
                linewidth=0.5,
                alpha=1,
                zorder=1000
            )
            dx = pred_traj[i, -1, 0] - pred_traj[i, -2, 0]
            dy = pred_traj[i, -1, 1] - pred_traj[i, -2, 1]
            ax.arrow3D(
                pred_traj[i, -1, 0], pred_traj[i, -1, 1], 0,
                dx, dy, 0,
                mutation_scale=5,
                linewidth=0.65,
                fc=color,
                ec=color,
                label='TempEns-LearnAgg trajs',
                zorder=1000
            )

    def plot_gt(self, ax, gt_traj):
        gt_traj = gt_traj.reshape(50, 2)
        color = 'red'
        ax.plot(
            gt_traj[:, 0], gt_traj[:, 1], zs=0, zdir='z',
            color=color,
            linestyle='dashed',
            linewidth=0.5,
            alpha=1,
            zorder=1000,
        )
        dx = gt_traj[-1, 0] - gt_traj[-2, 0]
        dy = gt_traj[-1, 1] - gt_traj[-2, 1]
        ax.arrow3D(
            gt_traj[-1, 0], gt_traj[-1, 1], 0,
            dx, dy, 0,
            mutation_scale=5,
            linewidth=0.5,
            arrowstyle="-|>",
            fc='red',
            ec='red',
            label='GT traj',
            zorder=1000
        )

    def matplot_traj(self, 
        ax, time, processed_data, 
        bsf_pred_trajs, bsw_pred_trajs, 
        tela_pred_trajs, gt_traj): 
        self.plot_history(ax, time, processed_data)
        self.plot_bsw_prediction(ax, bsw_pred_trajs)
        self.plot_bsf_prediction(ax, bsf_pred_trajs)
        self.plot_tela_prediction(ax, tela_pred_trajs)
        self.plot_gt(ax, gt_traj)
    

    def matplot_legend(self, ax):
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        legend = plt.legend(
            by_label.values(),
            by_label.keys(),
            fontsize=7.5,
            loc='upper right',
            facecolor='white',
        )
        legend.set_zorder(999)

    def forward(self, viz_dict) -> None:
        # 1. fetch data from dict
        processed_data = viz_dict['processed_data']
        bsf_pred_trajs = viz_dict['baseline_sf_pred_trajs']
        bsw_pred_trajs = viz_dict['baseline_sw_pred_trajs']
        tela_pred_trajs = viz_dict['tela_pred_trajs']
        gt_traj = viz_dict['gt_traj']
        dataset = viz_dict['dataset']
        
        # 2. preprocess visuzlize ingridient
        fig, ax = self.create_fig_and_ax(
                size_pixels=2000, 
                processed_data=processed_data, 
                gt_traj=gt_traj,
                sf_trajs=bsf_pred_trajs,
                sw_trajs=bsw_pred_trajs,
                k_trajs=tela_pred_trajs)
        map_data = self.preprocess_map(dataset, processed_data)
        time = 11
        agent_data, scenario_id = self.preprocess_agent(dataset, processed_data, time)

        # 3. plot scenario with prediction
        self.matplot_map(ax, map_data)
        self.matplot_agent(ax, agent_data)
        self.matplot_traj(
                ax, time, processed_data, 
                bsf_pred_trajs, 
                bsw_pred_trajs, 
                tela_pred_trajs, 
                gt_traj)
        self.matplot_legend(ax)
        
        # 4. save it to file 
        plt.savefig(f'./visualize_results/{scenario_id}.png', dpi=800)
        plt.clf()
        plt.close()
