import numpy as np

from sklearn.cluster import KMeans

import torch
from torch.nn import functional as F

def k_means(trajs, clsfs):
    endpoints = np.array([traj.detach().cpu().numpy()[-1] for traj in trajs])
    kmeans = KMeans(n_clusters=6, random_state=0, n_init='auto').fit(endpoints)

    k_means_trajs = []
    k_means_clsfs= []
    k_means_corresponding = []
    for i in range(6):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        cluster_trajs = torch.stack([trajs[j] for j in cluster_indices])
        cluster_clsfs = torch.stack([clsfs[j] for j in cluster_indices])
        k_means_traj = torch.mean(cluster_trajs, dim=0)
        k_means_clsf = torch.sum(cluster_clsfs, dim=0)
        k_means_trajs.append(k_means_traj)
        k_means_clsfs.append(k_means_clsf)
        k_means_corresponding.append(cluster_indices)

    k_means_trajs = torch.stack(k_means_trajs, dim=0)
    k_means_clsfs = F.softmax(torch.stack(k_means_clsfs, dim=0), dim=0)
    return k_means_trajs, k_means_clsfs, k_means_corresponding

