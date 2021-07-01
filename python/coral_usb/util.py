import numpy as np


def get_panorama_slices(width, n_split, overlap=True):
    if overlap:
        x_offsets = np.arange(n_split) * int(width / (n_split + 1))
        x_width = int(2 * width / (n_split + 1))
        x_offsets = x_offsets.astype(np.int)
        panorama_slices = [slice(x_offset, x_offset + x_width) for x_offset in x_offsets]
        panorama_slices.append(slice(width - int(x_width/2), int(x_width/2)))
        return panorama_slices
    else:
        x_offsets = np.arange(n_split) * int(width / n_split)
        x_width = int(width / n_split)
        x_offsets = x_offsets.astype(np.int)
        panorama_slices = [slice(x_offset, x_offset + x_width) for x_offset in x_offsets]
        return panorama_slices
