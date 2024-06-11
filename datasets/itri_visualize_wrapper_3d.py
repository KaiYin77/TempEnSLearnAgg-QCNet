import os
from pathlib import Path

import torch
import numpy as np
import pandas as pd 
import torch
import json
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d

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

class ITRIVisualizeWrapper3D:
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
    def split_lane_array(self, lane_array):
        return [lane_array[i:i + 10] for i in range(0, len(lane_array), 10)]

    def preprocess_map(self):
        with open('./raw_data/hd_map/waypoints.json', 'r') as f:
            waypoints_raw_data = json.load(f)
        waypoints = waypoints_raw_data['waypoints']
        lane_centerlines = []
        for wp in waypoints:
            points = wp['points']
            lane_array = np.array([[point['x'], point['y'], point['z']] for point in points])
            if len(lane_array) > 10:
                split_arrays = self.split_lane_array(lane_array)
                for array in split_arrays:
                    if len(array) == 10:
                        lane_centerlines.append(array)
            elif len(lane_array) == 10:
                lane_centerlines.append(lane_array)
        lane_centerlines = np.array(lane_centerlines)
        
        lane_polygons = []
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
            
            split_left_boundary_points = []
            split_right_boundary_points = []
            if len(left_boundary_points) > 10:
                split_left_arrays = self.split_lane_array(left_boundary_points)
                split_right_arrays = self.split_lane_array(right_boundary_points)
                for array in split_left_arrays:
                    if len(array) == 10:
                        split_left_boundary_points.append(array)
                for array in split_right_arrays:
                    if len(array) == 10:
                        split_right_boundary_points.append(array)
            elif len(left_boundary_points) == 10:
                split_left_boundary_points.append(left_boundary_points)
                split_right_boundary_points.append(right_boundary_points)

            for lp, rp in zip(split_left_boundary_points, split_right_boundary_points):
                lane_polygons.append(np.array(lp + rp[::-1] + [lp[0]]))
        lane_polygons = np.array(lane_polygons)

        with open('./raw_data/hd_map/pedestrian_crossing.json', 'r') as f:
            ped_cross_raw_data = json.load(f)
        ped_cross = ped_cross_raw_data['pedestrian_crossing']
        crosswalk_polygons = []
        for pc in ped_cross:
            points = pc['points']
            first_point = np.array([points[0]['x'], points[0]['y'], points[0]['z']])
            cross_array = np.array([[point['x'], point['y'], point['z']] for point in points])
            cross_array = np.vstack([cross_array, first_point])
            crosswalk_polygons.append(cross_array)
        crosswalk_polygons = np.array(crosswalk_polygons)
        
        map_data = {
            'lane_centerlines': lane_centerlines,
            'lane_polygons': lane_polygons,
            'crosswalk_polygons': crosswalk_polygons,
        }
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
                label='map',
            )
        lane_polygons = map_data['lane_polygons']
        for i, l_p in enumerate(lane_polygons):
            ax.plot(
                l_p[:, 0], l_p[:, 1], zs=0, zdir='z',
                color='dimgray',
                linewidth=0.5,
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

    def create_fig_and_ax(self, size_pixels):
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

        canva_center = [-6358, 3026]
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

    def forward(self, viz_dict=None) -> None:
        # 1. fetch data from dict
        #processed_data = viz_dict['processed_data']
        #bsf_pred_trajs = viz_dict['baseline_sf_pred_trajs']
        #bsw_pred_trajs = viz_dict['baseline_sw_pred_trajs']
        #tela_pred_trajs = viz_dict['tela_pred_trajs']
        #gt_traj = viz_dict['gt_traj']
        #dataset = viz_dict['dataset']
        
        # 2. preprocess visuzlize ingridient
        fig, ax = self.create_fig_and_ax(
                size_pixels=2000)
        map_data = self.preprocess_map()
        #time = 11
        #agent_data, scenario_id = self.preprocess_agent(dataset, processed_data, time)

        # 3. plot scenario with prediction
        self.matplot_map(ax, map_data)
        #self.matplot_agent(ax, agent_data)
        #self.matplot_traj(
        #        ax, time, processed_data, 
        #        bsf_pred_trajs, 
        #        bsw_pred_trajs, 
        #        tela_pred_trajs, 
        #        gt_traj)
        self.matplot_legend(ax)
        
        # 4. save it to file 
        plt.savefig(f'./visualize_results/only_map.png', dpi=800)
        plt.clf()
        plt.close()
