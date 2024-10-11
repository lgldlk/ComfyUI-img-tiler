import cv2
import numpy as np
import os
import sys
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
import math
import conf
from time import sleep
import random
import itertools
from collections import Counter

from decimal import Decimal, getcontext

# 设置精度
getcontext().prec = 28


# number of colors per image
COLOR_DEPTH = conf.COLOR_DEPTH
# image scale
IMAGE_SCALE = conf.IMAGE_SCALE
# tiles scales
RESIZING_SCALES = conf.RESIZING_SCALES
# number of pixels shifted to create each box (x,y)
PIXEL_SHIFT = conf.PIXEL_SHIFT
# multiprocessing pool size
POOL_SIZE = conf.POOL_SIZE
# if tiles can overlap
OVERLAP_TILES = conf.OVERLAP_TILES


# reduces the number of colors in an image
def color_quantization(img, n_colors):
    return np.round(img / 255 * n_colors) / n_colors * 255


# returns an image given its path
def read_image(path, mainImage=False):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img = color_quantization(img.astype('float'), COLOR_DEPTH)
    # scale the image according to IMAGE_SCALE, if this is the main image
    if mainImage:
        img = cv2.resize(img, (0, 0), fx=IMAGE_SCALE, fy=IMAGE_SCALE)
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
            if np.sum(x, dtype=np.int64) == 0:
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


# displays an image
def show_image(img, wait=True):
    cv2.imshow('img', img)
    if wait:
        cv2.waitKey(0)
    else:
        cv2.waitKey(1)


# load and process the tiles
def load_tiles(paths):
    print('Loading tiles')
    tiles = defaultdict(list)

    for path in paths:
        if os.path.isdir(path):
            for tile_name in tqdm(os.listdir(path)):
                tile = read_image(os.path.join(path, tile_name))
                mode,rel_freq = mode_color(tile, ignore_alpha=True)
                if mode is not None:
                    for scale in RESIZING_SCALES:
                        t = resize_image(tile, scale)
                        # res = tuple(t.shape[:2])
                        tiles[(
                            Decimal(tile.shape[0])*Decimal(str(scale)),
                           Decimal( tile.shape[1])*Decimal(str(scale))
                               )].append({
                            'tile': t,
                            'mode': mode,
                            "rel_freq":rel_freq 
                        })

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
            boxes.append({
                'img': img[y:y+box_height, x:x+box_width],
                'pos': (x, y),
                'size': (box_width, box_height) 
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
def get_processed_image_boxes(image_path, tiles):
    print('Getting and processing boxes')
    img = read_image(image_path, mainImage=True)  # Assuming `read_image` uses cv2.imread or similar method
    pool = Pool(POOL_SIZE)
    all_boxes = []
    x_coords = []
    y_coords = []

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
def place_tile(img, box):
    p1 = np.flip(box['pos'])
    p2 = p1 + box['img'].shape[:2]
    img_box = img[p1[0]:p2[0], p1[1]:p2[1]]
    
    mask = box['tile'][:, :, 3] != 0
    mask = mask[:img_box.shape[0], :img_box.shape[1]]
    if OVERLAP_TILES or not np.any(img_box[mask]):
        img_box[mask] = box['tile'][:img_box.shape[0], :img_box.shape[1], :][mask]

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

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    start_x = (new_width - target_width) // 2
    start_y = (new_height - target_height) // 2

    cropped_image = resized_image[start_y:(start_y + target_height), start_x:(start_x + target_width)]

    return cropped_image

# tiles the image
def create_tiled_image(boxes, res):
    print('Creating tiled image')
    img = np.zeros(shape=(res[0], res[1], 4), dtype=np.uint8)

    for box in tqdm(sorted(boxes, key=lambda x: x['min_dist'], reverse=OVERLAP_TILES)):
        place_tile(img, box)

    return img

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
    filtered_counts = max([count for count, occurrence in Counter(scale_count.values()).items() if occurrence > 1] or [0])
    if filtered_counts == 0:
        return None,None
    most_common_sizes = [size for size, count in scale_count.items() if count == filtered_counts]

    def is_multiple_of_any(size, sizes):
        return all((size[0] % s[0] == 0 and size[1] % s[1] == 0) or (s[0] % size[0] == 0 and s[1] % size[1] == 0) for s in sizes)

    most_common_sizes = list(filter(lambda x: is_multiple_of_any(x, most_common_sizes), tile_keys))
    non_multiple_sizes = list(set(tile_keys) - set(most_common_sizes))

    return most_common_sizes, non_multiple_sizes
# main
def main():
    print(sys.argv,len(sys.argv))
    image_path='images/test.jpg'
    tiles_paths=['./tiles/lego/gen_lego_v']

    if len(sys.argv) >= 3:
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
        else:
            image_path = conf.IMAGE_TO_TILE

        if len(sys.argv) > 2:
            tiles_paths = sys.argv[2:]
        else:
            tiles_paths = conf.TILES_FOLDER.split(' ')

    if not os.path.exists(image_path):
        print('Image not found')
        exit(-1)
    for path in tiles_paths:
        if not os.path.exists(path):
            print('Tiles folder not found')
            exit(-1)
    tiles = load_tiles(tiles_paths)
    tile_keys = list(tiles.keys())
    
    print(f'Tiles loaded: {len(tile_keys)} sizes: {tile_keys}')
    most_common_size, non_multiple_sizes = find_most_common_scale(tile_keys)
    if most_common_size is not None:
        for point_b in non_multiple_sizes:
            closest_point = min(
                    [point_a for point_a in most_common_size if point_a < point_b],
                    key=lambda point_a: euclidean_distance(point_b, point_a)
                )
            print(f"Closest point to {point_b} is {closest_point}")
            # 调整 tiles  point_b to closest_point
            for tile in tiles[point_b]:
                tile['tile'] =resize_and_crop(tile['tile'], closest_point[1], closest_point[0])
                mode, rel_freq = mode_color(tile['tile'], ignore_alpha=True)
                tile['mode'] = mode
                tile['rel_freq'] = rel_freq
            tiles[closest_point] += tiles.pop(point_b)
    print(list(tiles.keys()))
    boxes, original_res = get_processed_image_boxes(image_path, tiles)
    img = create_tiled_image(boxes, original_res, )
    

    cv2.imwrite(conf.OUT, img)


if __name__ == "__main__":
    main()
