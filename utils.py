import itertools
import os
from collections import defaultdict
import cv2
import sys
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool
import math
import torch
from PIL import Image, ImageEnhance

from . import conf

# import conf
from collections import Counter

import comfy.utils
from decimal import Decimal, getcontext

# number of colors per image
COLOR_DEPTH = conf.COLOR_DEPTH


# number of pixels shifted to create each box (x,y)
PIXEL_SHIFT = conf.PIXEL_SHIFT
# multiprocessing pool size
POOL_SIZE = conf.POOL_SIZE


def get_all_directories(path):
    directories = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            directories.append(os.path.normpath(os.path.join(root, dir)))
    return directories


def find_files_with_extensions(directory, extensions, exclude_strs=None):
    result = []
    for root, dirs, files in os.walk(directory):
        if exclude_strs and any(exclude_str in root for exclude_str in exclude_strs):
            continue
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                result.append(os.path.join(root, file))
    return result


# returns an image given its path
def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img = color_quantization(img.astype("float"), COLOR_DEPTH)
    return img.astype("uint8")


# scales an image
def resize_image(img, ratio):
    img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))
    return img


# the most frequent color in an image and its relative frequency
def mode_color(img, ignore_alpha=False):
    counter = defaultdict(int)
    total = 0
    for y in img:
        for x in y:
            if sum(x) == 0:
                continue
            if len(x) < 4 or ignore_alpha or x[3] != 0:
                counter[tuple(x[:3])] += 1
            else:
                counter[(-1, -1, -1)] += 1
            total += 1

    if total > 0:
        mode_color = max(counter, key=counter.get)
        if mode_color == (-1, -1, -1):
            return None, None
        else:
            return mode_color, counter[mode_color] / total
    else:
        return None, None


# load and process the tiles
def load_tiles(paths, resizing_scales):
    print("Loading tiles")
    tiles = defaultdict(list)

    for path in paths:
        if os.path.isdir(path):
            for tile_name in tqdm(os.listdir(path)):
                tile = read_image(os.path.join(path, tile_name))
                mode, rel_freq = mode_color(tile, ignore_alpha=True)
                if mode is not None:
                    for scale in resizing_scales:
                        t = resize_image(tile, scale)
                        # res = tuple(t.shape[:2])
                        tiles[
                            (
                                Decimal(tile.shape[0]) * Decimal(str(scale)),
                                Decimal(tile.shape[1]) * Decimal(str(scale)),
                            )
                        ].append({"tile": t, "mode": mode, "rel_freq": rel_freq})

    return tiles


# returns the boxes (image and start pos) from an image, with 'res' resolution
def image_boxes(img, res):
    if not PIXEL_SHIFT:
        shift = np.flip(res)
    else:
        shift = PIXEL_SHIFT

    boxes = []
    img_height, img_width = img.shape[:2]
    w_shift, h_shift = (int(shift[0]), int(shift[1]))
    for y in range(0, img_height, h_shift):
        for x in range(0, img_width, w_shift):
            box_height = int(res[0])  # Handle edge cases for height
            box_width = int(res[1])  # Handle edge cases for width
            boxes.append(
                {
                    "img": img[y : y + box_height, x : x + box_width],
                    "pos": (x, y),
                    "size": (box_width, box_height),
                }
            )

    return boxes


# euclidean distance between two colors
def color_distance(c1, c2):
    c1_int = [int(x) for x in c1]
    c2_int = [int(x) for x in c2]
    return math.sqrt(
        (c1_int[0] - c2_int[0]) ** 2
        + (c1_int[1] - c2_int[1]) ** 2
        + (c1_int[2] - c2_int[2]) ** 2
    )


# returns the most similar tile to a box (in terms of color)
def most_similar_tile(box_mode_freq, tiles):
    if not box_mode_freq[0]:
        return (0, np.zeros(shape=tiles[0]["tile"].shape))
    else:
        min_distance = None
        min_tile_img = None
        for t in tiles:
            dist = (1 + color_distance(box_mode_freq[0], t["mode"])) / box_mode_freq[1]
            if min_distance is None or dist < min_distance:
                min_distance = dist
                min_tile_img = t["tile"]
        return (min_distance, min_tile_img)


# builds the boxes and finds the best tile for each one
def get_processed_image_boxes(img, tiles):
    print("Getting and processing boxes")
    pool = Pool(POOL_SIZE)
    all_boxes = []

    for res, ts in tqdm(sorted(tiles.items(), reverse=True)):
        boxes = image_boxes(img, res)
        modes = pool.map(mode_color, [x["img"] for x in boxes])
        most_similar_tiles = pool.starmap(
            most_similar_tile, zip(modes, [ts for x in range(len(modes))])
        )

        i = 0
        for min_dist, tile in most_similar_tiles:
            boxes[i]["min_dist"] = min_dist
            boxes[i]["tile"] = tile
            i += 1

        all_boxes += boxes

    return all_boxes, img.shape


# places a tile in the image
def place_tile(img, box, overlap_tiles):
    p1 = np.flip(box["pos"])
    p2 = p1 + box["img"].shape[:2]
    img_box = img[p1[0] : p2[0], p1[1] : p2[1]]
    mask = box["tile"][:, :, 3] != 0
    mask = mask[: img_box.shape[0], : img_box.shape[1]]
    if overlap_tiles or not np.any(img_box[mask]):
        img_box[mask] = box["tile"][: img_box.shape[0], : img_box.shape[1], :][mask]


def resize_and_crop(image, target_width, target_height):
    if not isinstance(target_width, int):
        target_width = int(round(target_width))
    if not isinstance(target_height, int):
        target_height = int(round(target_height))
    original_height, original_width = image.shape[:2]
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    scale = max(scale_w, scale_h)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )

    start_x = (new_width - target_width) // 2
    start_y = (new_height - target_height) // 2

    cropped_image = resized_image[
        start_y : (start_y + target_height), start_x : (start_x + target_width)
    ]

    return cropped_image


# tiles the image
def create_tiled_image(boxes, res, overlap_tiles):
    print("Creating tiled image")
    img = np.zeros(shape=(res[0], res[1], 4), dtype=np.uint8)

    for box in tqdm(sorted(boxes, key=lambda x: x["min_dist"], reverse=overlap_tiles)):
        place_tile(img, box, overlap_tiles)

    return img


def merge_tiles(tiles1, tiles2):
    merged_tiles = defaultdict(
        list, tiles1
    )  # 创建一个默认从 tiles1 初始化的 defaultdict

    for key, value in tiles2.items():
        merged_tiles[key].extend(value)
    return merged_tiles


def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def find_most_common_scale(tile_keys):
    scale_count = defaultdict(int)
    non_multiple_sizes = []

    for size1, size2 in itertools.combinations(tile_keys, 2):
        h1, w1 = size1
        h2, w2 = size2
        if (h2 % h1 == 0 and w2 % w1 == 0) or (h1 % h2 == 0 and w1 % w2 == 0):
            scale_count[size1] += 1
            scale_count[size2] += 1
    filtered_counts = max(
        [
            count
            for count, occurrence in Counter(scale_count.values()).items()
            if occurrence > 1
        ]
        or [0]
    )
    if filtered_counts == 0:
        return None, None
    most_common_sizes = [
        size for size, count in scale_count.items() if count == filtered_counts
    ]

    def is_multiple_of_any(size, sizes):
        return all(
            (size[0] % s[0] == 0 and size[1] % s[1] == 0)
            or (s[0] % size[0] == 0 and s[1] % size[1] == 0)
            for s in sizes
        )

    most_common_sizes = list(
        filter(lambda x: is_multiple_of_any(x, most_common_sizes), tile_keys)
    )
    non_multiple_sizes = list(set(tile_keys) - set(most_common_sizes))

    return most_common_sizes, non_multiple_sizes


# reduces the number of colors in an image
def color_quantization(img, n_colors):
    return np.round(img / 255 * n_colors) / n_colors * 255


def torch_tensor_to_cv2_image(tensor_image):
    # 1. 去掉 batch 维度 (假设输入形状为 [batch, height, width, channels])
    tensor_image = tensor_image.squeeze(0)

    # 2. 将归一化的 [0, 1] 数据转换为 [0, 255]
    tensor_image = (tensor_image * 255).clamp(0, 255).byte()

    # 3. 将 PyTorch tensor 转换为 numpy 数组
    numpy_image = tensor_image.numpy()

    # 4. 确保通道顺序为 (height, width, channels)，如果输入格式是 (channels, height, width)，则调整维度
    if numpy_image.shape[0] in [1, 3]:  # 通道在第一维
        numpy_image = np.transpose(
            numpy_image, (1, 2, 0)
        )  # 转换为 (height, width, channels)

    # 5. 如果是 RGB 图像，OpenCV 需要 BGR 格式
    if numpy_image.shape[2] == 3:  # 检查是否为 RGB 图像
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2BGRA)

    return numpy_image


def cv2_image_to_torch_tensor(cv2_image):
    modified_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2BGRA)

    modified_image = modified_image.astype(np.uint8)
    modified_image = modified_image / 255
    modified_image = torch.from_numpy(modified_image).unsqueeze(0)

    return modified_image


def add_alpha_channel(rgb_tensor, alpha_value=1.0):
    batch_size, height, width, channels = rgb_tensor.shape
    alpha_channel = torch.full((batch_size, height, width, 1), alpha_value)
    rgba_tensor = torch.cat((rgb_tensor, alpha_channel), dim=-1)
    return rgba_tensor


def composite(destination, source, x, y, mask=None, multiplier=8, resize_source=False):
    source = source.to(destination.device)
    if resize_source:
        source = torch.nn.functional.interpolate(
            source, size=(destination.shape[2], destination.shape[3]), mode="bilinear"
        )

    source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])

    x = max(-source.shape[3] * multiplier, min(x, destination.shape[3] * multiplier))
    y = max(-source.shape[2] * multiplier, min(y, destination.shape[2] * multiplier))

    left, top = (x // multiplier, y // multiplier)
    right, bottom = (
        left + source.shape[3],
        top + source.shape[2],
    )

    if mask is None:
        mask = torch.ones_like(source)
    else:
        mask = mask.to(destination.device, copy=True)
        mask = torch.nn.functional.interpolate(
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
            size=(source.shape[2], source.shape[3]),
            mode="bilinear",
        )
        mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])

    # calculate the bounds of the source that will be overlapping the destination
    # this prevents the source trying to overwrite latent pixels that are out of bounds
    # of the destination
    visible_width, visible_height = (
        destination.shape[3] - left + min(0, x),
        destination.shape[2] - top + min(0, y),
    )

    mask = mask[:, :, :visible_height, :visible_width]
    inverse_mask = torch.ones_like(mask) - mask

    source_portion = mask * source[:, :, :visible_height, :visible_width]
    destination_portion = inverse_mask * destination[:, :, top:bottom, left:right]

    destination[:, :, top:bottom, left:right] = source_portion + destination_portion
    return destination
