import numpy as np


def get_panorama_slices(width, n_split, overlap=True):
    if overlap:
        x_offsets = np.arange(n_split) * int(width / (n_split + 1))
        x_width = int(2 * width / (n_split + 1))
    else:
        x_offsets = np.arange(n_split) * int(width / n_split)
        x_width = int(width / n_split)
    x_offsets = x_offsets.astype(np.int)
    return [slice(x_offset, x_offset + x_width if x_offset + x_width <= width else x_offset + x_width - width) for x_offset in x_offsets]
