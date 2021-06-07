import os
import numpy as np

idxpath = 'label_feat_list.txt'
featidx = np.loadtxt(idxpath, dtype=int, delimiter=' ')

def load_features(path):
    if os.path.exists(path):
        data = np.loadtxt(path, dtype=float, delimiter=' ')

        return data[featidx]
    else:
        return None
