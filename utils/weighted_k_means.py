import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.nn import functional as F

def weighted_k_means(trajs, clsfs):
    endpoints = np.array([traj.detach().cpu().numpy()[-1] for traj in trajs])
    kmeans = KMeans(n_clusters=6, random_state=0, n_init='auto').fit(endpoints)

    k_means_trajs = []
    k_means_clsfs = []

    for i in range(6):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        cluster_trajs = torch.stack([trajs[j] for j in cluster_indices])
        cluster_weights = torch.stack([clsfs[j] for j in cluster_indices])

        # Compute the weighted mean for trajectories within the cluster
        weighted_cluster_traj = torch.sum(cluster_trajs * cluster_weights.view(-1, 1, 1), dim=0) / torch.sum(cluster_weights)

        # Combine the classification scores for trajectories within the cluster
        weighted_cluster_clsfs = torch.sum(cluster_weights, dim=0)

        k_means_trajs.append(weighted_cluster_traj)
        k_means_clsfs.append(weighted_cluster_clsfs)

    k_means_trajs = torch.stack(k_means_trajs, dim=0)
    
    # Apply softmax to the weighted classification scores to get probabilities
    k_means_clsfs = F.softmax(torch.stack(k_means_clsfs, dim=0), dim=0)
    
    return k_means_trajs, k_means_clsfs

# Example usage:
# k_means_trajs, k_means_clsfs = weighted_k_means(trajs, clsfs)

