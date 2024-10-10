
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
from  .  import conf


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
  
  

# returns an image given its path
def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img = color_quantization(img.astype('float'), COLOR_DEPTH)
    return img.astype('uint8')


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
                counter[(-1,-1,-1)] += 1
            total += 1

    if total > 0:
        mode_color = max(counter, key=counter.get)
        if mode_color == (-1,-1,-1):
            return None, None
        else:
            return mode_color, counter[mode_color] / total
    else:
        return None, None

# load and process the tiles
def load_tiles(paths,resizing_scales):
    print('Loading tiles')
    tiles = defaultdict(list)

    for path in paths:
        if os.path.isdir(path):
            for tile_name in tqdm(os.listdir(path)):
                tile = read_image(os.path.join(path, tile_name))
                mode, rel_freq = mode_color(tile, ignore_alpha=True)
                if mode is not None:
                    for scale in resizing_scales:
                        t = resize_image(tile, scale)
                        res = tuple(t.shape[:2])
                        tiles[res].append({
                            'tile': t,
                            'mode': mode,
                            'rel_freq': rel_freq
                        })

    return tiles


# returns the boxes (image and start pos) from an image, with 'res' resolution
def image_boxes(img, res):
    if not PIXEL_SHIFT:
        shift = np.flip(res)
    else:
        shift = PIXEL_SHIFT

    boxes = []
    for y in range(0, img.shape[0], shift[1]):
        for x in range(0, img.shape[1], shift[0]):
            boxes.append({
                'img': img[y:y+res[0], x:x+res[1]],
                'pos': (x,y)
            })

    return boxes


# euclidean distance between two colors
def color_distance(c1, c2):
    c1_int = [int(x) for x in c1]
    c2_int = [int(x) for x in c2]
    return math.sqrt((c1_int[0] - c2_int[0])**2 + (c1_int[1] - c2_int[1])**2 + (c1_int[2] - c2_int[2])**2)


# returns the most similar tile to a box (in terms of color)
def most_similar_tile(box_mode_freq, tiles):
    if not box_mode_freq[0]:
        return (0, np.zeros(shape=tiles[0]['tile'].shape))
    else:
        min_distance = None
        min_tile_img = None
        for t in tiles:
            dist = (1 + color_distance(box_mode_freq[0], t['mode'])) / box_mode_freq[1]
            if min_distance is None or dist < min_distance:
                min_distance = dist
                min_tile_img = t['tile']
        return (min_distance, min_tile_img)


# builds the boxes and finds the best tile for each one
def get_processed_image_boxes(img, tiles):
    print('Getting and processing boxes')
    pool = Pool(POOL_SIZE)
    all_boxes = []

    for res, ts in tqdm(sorted(tiles.items(), reverse=True)):
        boxes = image_boxes(img, res)
        modes = pool.map(mode_color, [x['img'] for x in boxes])
        most_similar_tiles = pool.starmap(most_similar_tile, zip(modes, [ts for x in range(len(modes))]))

        i = 0
        for min_dist, tile in most_similar_tiles:
            boxes[i]['min_dist'] = min_dist
            boxes[i]['tile'] = tile
            i += 1

        all_boxes += boxes

    return all_boxes, img.shape


# places a tile in the image
def place_tile(img, box,overlap_tiles):
    p1 = np.flip(box['pos'])
    p2 = p1 + box['img'].shape[:2]
    img_box = img[p1[0]:p2[0], p1[1]:p2[1]]
    mask = box['tile'][:, :, 3] != 0
    mask = mask[:img_box.shape[0], :img_box.shape[1]]
    if overlap_tiles or not np.any(img_box[mask]):
        img_box[mask] = box['tile'][:img_box.shape[0], :img_box.shape[1], :][mask]


# tiles the image
def create_tiled_image(boxes, res,overlap_tiles):
    print('Creating tiled image')
    img = np.zeros(shape=(res[0], res[1], 4), dtype=np.uint8)

    for box in tqdm(sorted(boxes, key=lambda x: x['min_dist'], reverse=overlap_tiles)):
        place_tile(img, box,overlap_tiles)

    return img

def merge_tiles(tiles1, tiles2):
    merged_tiles = defaultdict(list, tiles1)  # 创建一个默认从 tiles1 初始化的 defaultdict

    for key, value in tiles2.items():
        merged_tiles[key].extend(value)  # 如果键相同，则扩展列表
    
    return merged_tiles




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
        numpy_image = np.transpose(numpy_image, (1, 2, 0))  # 转换为 (height, width, channels)

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