import cv2
import numpy as np
import os
import sys
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
import math
from  .  import conf
from .utils import torch_tensor_to_cv2_image,color_quantization,cv2_image_to_torch_tensor,get_all_directories
import re
import json

# number of colors per image
COLOR_DEPTH = conf.COLOR_DEPTH


# number of pixels shifted to create each box (x,y)
PIXEL_SHIFT = conf.PIXEL_SHIFT
# multiprocessing pool size
POOL_SIZE = conf.POOL_SIZE






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


# # main
# def main():
#     if len(sys.argv) > 1:
#         image_path = sys.argv[1]
#     else:
#         image_path = conf.IMAGE_TO_TILE

#     if len(sys.argv) > 2:
#         tiles_paths = sys.argv[2:]
#     else:
#         tiles_paths = conf.TILES_FOLDER.split(' ')

#     if not os.path.exists(image_path):
#         print('Image not found')
#         exit(-1)
#     for path in tiles_paths:
#         if not os.path.exists(path):
#             print('Tiles folder not found')
#             exit(-1)

#     tiles = load_tiles(tiles_paths)
#     boxes, original_res = get_processed_image_boxes(image_path, tiles)
#     img = create_tiled_image(boxes, original_res, render=conf.RENDER)
#     cv2.imwrite(conf.OUT, img)


# if __name__ == "__main__":
#     main()

now_folder_path = os.path.dirname(__file__)


tilesList=[]
for dir in get_all_directories(
    os.path.join(now_folder_path,"tiles")
):
  dir_res=re.search(r"[\\|\/]gen_(.*)",dir)
  if dir_res:
    tilesList.append({
      "name":dir_res.group(1),
      "path":dir
    })


class TilerSelect:
  @classmethod
  def INPUT_TYPES(cls):
    select_tile_options=[]
    for tile in tilesList:
      select_tile_options.append(tile["name"])
    return {
      "required": {
        "tile": (select_tile_options,
                          {"default": "square_50"}),
      }
    }
  RETURN_TYPES = ("SelectTile",)
  RETURN_NAMES = ("tile",)
  OUTPUT_IS_LIST = (False,)
  FUNCTION = "run_tile"
  CATEGORY = "😱 PointAgiClub"
  def run_tile(self,tile):
    result = next((x for x in tilesList if x["name"] == tile), None)
    return (result,)
    
class TilerImage:
  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "image":("IMAGE",),
        "overlap_tiles":("BOOLEAN",{
            "default":False
        }),
        # RESIZING_SCALES
        "resizing_scales":("STRING",{
            "default":"[0.5, 0.4, 0.3, 0.2, 0.1]"
        }),
        "tile1":("SelectTile",),
      }
    }
  RETURN_TYPES = ("IMAGE",)
  RETURN_NAMES = ("image",)
  OUTPUT_IS_LIST = (False,)
  FUNCTION = "create_tiled_image"
  CATEGORY = "😱 PointAgiClub"
  def create_tiled_image(self,image,overlap_tiles,resizing_scales,**kwargs):
    tiles = []

    for k, v in kwargs.items():
            tiles.append(v['path'])
    image = torch_tensor_to_cv2_image(image)
    try:
      resizing_scales = json.loads(resizing_scales)
      for scale in resizing_scales:
        if not isinstance(scale, float):
          raise ValueError("Invalid resizing_scales, required a list of floats")
    except:
        raise ValueError("Invalid resizing_scales, required a list of floats")
    
    
    tiles = load_tiles(tiles,resizing_scales)
    boxes, original_res = get_processed_image_boxes(image, tiles)
    img = create_tiled_image(boxes, original_res,overlap_tiles)
    res=cv2_image_to_torch_tensor(img)
    return (res,)

