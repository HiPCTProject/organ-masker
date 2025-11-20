import numpy as np
from tqdm import tqdm
from scipy.ndimage import label, binary_opening

def fill_2d_holes(volume):
    """
    Slice-by-slice 2D hole filling using border-connected BG suppression.
    Operates on uint8 [0,1] volume; returns uint8 [0,1].
    """
    result = np.zeros_like(volume, dtype=np.uint8)
    for i in tqdm(range(volume.shape[0]), desc="2D Hole Filling", unit="slice"):
        slice_ = volume[i].astype(bool)
        slice_ = binary_opening(slice_, structure=np.ones((20, 20)))
        inverse = ~slice_
        labeled, _ = label(inverse)
        border_labels = np.unique(np.concatenate([
            labeled[0, :].ravel(), labeled[-1, :].ravel(),
            labeled[:, 0].ravel(), labeled[:, -1].ravel()
        ]))
        background_mask = np.isin(labeled, border_labels)
        filled = np.where(background_mask, 0, 1).astype(np.uint8)
        result[i] = np.maximum(slice_.astype(np.uint8), filled)
    return result
