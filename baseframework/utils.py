import numpy as np

def distance_calc(pos1, pos2, dist="euclidean", dim_weight=None):
    if dim_weight is None:
        dim_weight = np.ones(pos1.shape[-1])
    if dist == "euclidean":
        if len(pos1.shape) == 1 or len(pos2.shape) == 1:
            return np.sqrt(np.sum(dim_weight * np.square(pos1 - pos2), axis=-1))
        else:
            return np.sqrt(np.sum(dim_weight * np.square(pos1[:, None, :] - pos2[None, :, :]), axis=-1))
    else:
        raise NotImplementedError

