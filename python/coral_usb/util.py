import numpy as np


def get_panorama_slices(width, n_split, overlap=True):
    if overlap:
        x_offsets = np.arange(n_split) * int(width / (n_split + 1))
        x_width = int(2 * width / (n_split + 1))
        x_offsets = x_offsets.astype(np.int)
        panorama_slices = [
            slice(x_offset, x_offset + x_width) for x_offset in x_offsets]
        panorama_slices.append(
            slice(width - int(x_width / 2), int(x_width / 2)))
        return panorama_slices
    else:
        x_offsets = np.arange(n_split) * int(width / n_split)
        x_width = int(width / n_split)
        x_offsets = x_offsets.astype(np.int)
        panorama_slices = [
            slice(x_offset, x_offset + x_width) for x_offset in x_offsets]
        return panorama_slices


def get_panorama_sliced_image(panorama_img, panorama_slice):
    if panorama_slice.start > panorama_slice.stop:
        left_sliced_img = panorama_img[:, panorama_slice.start:, :]
        right_sliced_img = panorama_img[:, :panorama_slice.stop, :]
        img = np.concatenate([left_sliced_img, right_sliced_img], 1)
    else:
        img = panorama_img[:, panorama_slice, :]
    return img


def get_tile_slices(height, width, overlap=True,
                    tile_sizes=((300, 300), (250, 250)),
                    tile_overlap_rate=0.1):
    tile_slices = []
    for tile_size in tile_sizes:
        tile_height, tile_width = tile_size
        if overlap:
            tile_height_overlap = int(tile_height * tile_overlap_rate)
            tile_width_overlap = int(tile_width * tile_overlap_rate)
        else:
            tile_height_overlap = 0
            tile_width_overlap = 0
        h_stride = tile_height - tile_height_overlap
        w_stride = tile_width - tile_width_overlap
        for y_min in range(0, height, h_stride):
            for x_min in range(0, width, w_stride):
                y_max = min(height, y_min + tile_height)
                x_max = min(width, x_min + tile_width)
                tile_slices.append((slice(y_min, y_max), slice(x_min, x_max)))
    return tile_slices


def get_tile_sliced_image(img, tile_slice):
    sliced_img = img[tile_slice[0], tile_slice[1], :]
    return sliced_img


# copied from chainercv
def non_maximum_suppression(bbox, thresh, score=None, limit=None):
    if len(bbox) == 0:
        return np.zeros((0,), dtype=np.int32)

    if score is not None:
        order = score.argsort()[::-1]
        bbox = bbox[order]
    bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)

    selec = np.zeros(bbox.shape[0], dtype=bool)
    for i, b in enumerate(bbox):
        tl = np.maximum(b[:2], bbox[selec, :2])
        br = np.minimum(b[2:], bbox[selec, 2:])
        area = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)

        iou = area / (bbox_area[i] + bbox_area[selec] - area)
        if (iou >= thresh).any():
            continue

        selec[i] = True
        if limit is not None and np.count_nonzero(selec) >= limit:
            break

    selec = np.where(selec)[0]
    if score is not None:
        selec = order[selec]
    return selec.astype(np.int32)


# copied from chainercv
def generate_random_bbox(n, img_size, min_length, max_length):
    H, W = img_size
    y_min = np.random.uniform(0, H - max_length, size=(n,))
    x_min = np.random.uniform(0, W - max_length, size=(n,))
    y_max = y_min + np.random.uniform(min_length, max_length, size=(n,))
    x_max = x_min + np.random.uniform(min_length, max_length, size=(n,))
    bbox = np.stack((y_min, x_min, y_max, x_max), axis=1).astype(np.int)
    return bbox


def generate_random_point(n_key, bbox):
    point = []
    for bb in bbox:
        y_min, x_min, y_max, x_max = bb
        key_y = np.random.randint(y_min, y_max, size=(n_key, ))
        key_x = np.random.randint(x_min, x_max, size=(n_key, ))
        point.append(list(zip(key_y, key_x)))
    point = np.array(point, dtype=np.int)
    return point


def generate_random_label(img_size, label_ids):
    label = np.random.randint(0, len(label_ids), size=img_size)
    return label
