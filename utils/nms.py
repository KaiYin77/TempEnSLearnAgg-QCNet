import numpy as np

def pairwise_distance(input_1, input_2):
    return np.linalg.norm(input_1 - input_2)

def non_diverse_rule(max_conf_traj, pred_selected, pred, radius=2.5):
    if pairwise_distance(pred_selected[-1], pred[-1]) < radius:
        return True
    return False

def nms(out, clfs, tgt_selection=6, radius=2.5):
    index = np.argsort(clfs, axis=0)
    select_by_nms = []
    candidate_index = index.tolist()
    max_conf_index = candidate_index[-1]
    max_conf_traj = out[max_conf_index]
    
    while len(candidate_index) > 0:
        select_by_nms.append(candidate_index.pop())
        selected = select_by_nms[-1]
        to_remove = []
        for idx in candidate_index:
            if non_diverse_rule(max_conf_traj, out[selected], out[idx], radius=radius):
                to_remove.append(idx)
        for idx in to_remove:
            candidate_index.remove(idx)
        if len(select_by_nms) == tgt_selection:
            break
    
    idxs = index[::-1].tolist()
    i = 0
    while len(select_by_nms) < tgt_selection:
        idx = idxs[i]
        if idx not in select_by_nms:
            select_by_nms.append(idx)
        i += 1
    assert len(select_by_nms) == tgt_selection
    return out[select_by_nms], clfs[select_by_nms]

