import cv2
import os
from collections import defaultdict
from .utils import torch_tensor_to_cv2_image,color_quantization,cv2_image_to_torch_tensor,get_all_directories,merge_tiles,get_processed_image_boxes,create_tiled_image,load_tiles,mode_color,resize_image,find_most_common_scale,euclidean_distance,resize_and_crop,composite
import re
import json
import numpy as np

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

default_resizing_scales="[0.8, 0.4, 0.2, 0.1]"
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
        "resizing_scales":("STRING",{
            "default":default_resizing_scales
        }),
      }
    }
  RETURN_TYPES = ("Pc_Tiles",)
  RETURN_NAMES = ("tile",)
  OUTPUT_IS_LIST = (False,)
  FUNCTION = "run_tile"
  CATEGORY = "😱 PointAgiClub"
  def run_tile(self,tile,resizing_scales):
    result = next((x for x in tilesList if x["name"] == tile), None)
    try:
      resizing_scales = json.loads(resizing_scales)
      for scale in resizing_scales:
        if not isinstance(scale, float):
          raise ValueError("Invalid resizing_scales, required a list of floats")
    except:
        raise ValueError("Invalid resizing_scales, required a list of floats")
    
    tiles = load_tiles([
        result['path']
        ],resizing_scales)
    return (tiles,)
    
class TilerImage:
  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "image":("IMAGE",),
        "overlap_tiles":("BOOLEAN",{
            "default":False
        }),
        "auto_perfect_grid":("BOOLEAN",{
            "default":True
        }),

        "tile1":("Pc_Tiles",),
      }
    }
  RETURN_TYPES = ("IMAGE",)
  RETURN_NAMES = ("image",)
  OUTPUT_IS_LIST = (False,)
  FUNCTION = "create_tiled_image"
  CATEGORY = "😱 PointAgiClub"
  def create_tiled_image(self,image,overlap_tiles,auto_perfect_grid,**kwargs):
    tiles = defaultdict(list)
    for k, v in kwargs.items():
            tiles=merge_tiles(tiles,v)
    tile_keys = list(tiles.keys())
    
    print(f'Tiles loaded: {len(tile_keys)} sizes: {tile_keys}')
    if auto_perfect_grid:
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

    image = torch_tensor_to_cv2_image(image)
    boxes, original_res = get_processed_image_boxes(image, tiles)
    img = create_tiled_image(boxes, original_res,overlap_tiles)
    res=cv2_image_to_torch_tensor(img)
    return (res,)



class TileMaker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth": ("INT", {"default": 4, "min": 2, "max": 16}),
                "rotations": ("STRING", {"default": "[0]"}),
                "resizing_scales":("STRING",{
                    "default":default_resizing_scales
                }),
                # "output_folder": ("STRING", {"default": "generated_images"}),
                
            }
        }

    RETURN_TYPES = ("Pc_Tiles",)
    RETURN_NAMES = ("tile",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "generate_images"
    CATEGORY =  "😱 PointAgiClub"
    def generate_images(self, image, depth, rotations,resizing_scales, 
                        # output_folder
                        ):
        try:
            rotations = json.loads(rotations)
            if not all(isinstance(rotation, (int, float)) for rotation in rotations):
                raise ValueError("Invalid rotations format, should be a list of numbers")
        except:
            raise ValueError("Invalid rotations format, should be a JSON formatted list of numbers")
        try:
            resizing_scales = json.loads(resizing_scales)
            for scale in resizing_scales:
                if not isinstance(scale, float):
                    raise ValueError("Invalid resizing_scales, required a list of floats")
        except:
            raise ValueError("Invalid resizing_scales, required a list of floats")
        img = torch_tensor_to_cv2_image(image)
        img = img.astype('float')
        height, width, channels = img.shape
        center = (width / 2, height / 2)
        tiles = defaultdict(list)
        # Iterate through color multipliers
        for b in np.linspace(0, 1, depth):
            for g in np.linspace(0, 1, depth):
                for r in np.linspace(0, 1, depth):
                    mult_vector = [b, g, r]
                    if channels == 4:
                        mult_vector.append(1)
                    
                    new_img = (img * mult_vector).astype('uint8')
                    
                    for rotation in rotations:
                        rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1)
                        abs_cos = abs(rotation_matrix[0, 0])
                        abs_sin = abs(rotation_matrix[0, 1])
                        new_w = int(height * abs_sin + width * abs_cos)
                        new_h = int(height * abs_cos + width * abs_sin)
                        rotation_matrix[0, 2] += new_w / 2 - center[0]
                        rotation_matrix[1, 2] += new_h / 2 - center[1]

                        rotated_img = cv2.warpAffine(new_img, rotation_matrix, (new_w, new_h))
                        mode, rel_freq = mode_color(rotated_img, ignore_alpha=(channels == 4))

                        if mode is not None:
                            for scale in resizing_scales:
                                resized_tile = resize_image(rotated_img, scale)
                                res = tuple(resized_tile.shape[:2])

                                tiles[res].append({
                                    'tile': resized_tile,
                                    'mode': mode,
                                    'rel_freq': rel_freq
                                })

                        
                        # img_name = f"{output_folder}/image_{round(r, 1)}_{round(g, 1)}_{round(b, 1)}_r{rotation}.png"
                        # cv2.imwrite(img_name, rotated_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                        # generated_images.append(img_name)

        return (tiles,)
    
    
    

class ImageListTileMaker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "rotations": ("STRING", {"default": "[0]"}),
                "resizing_scales":("STRING",{
                    "default":default_resizing_scales
                }),
                # "output_folder": ("STRING", {"default": "generated_images"}),
                
            }
        }
    INPUT_IS_LIST = True
    RETURN_TYPES = ("Pc_Tiles",)
    RETURN_NAMES = ("tile",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "generate_images"
    CATEGORY =  "😱 PointAgiClub"
    
    def generate_images(self, image, rotations,resizing_scales, 
                        # output_folder
                        ):

        try:
            rotations = json.loads(rotations[0])
            if not all(isinstance(rotation, (int, float)) for rotation in rotations):
                raise ValueError("Invalid rotations format, should be a list of numbers")
        except:
            raise ValueError("Invalid rotations format, should be a JSON formatted list of numbers")
        try:
            resizing_scales = json.loads(resizing_scales[0])
            for scale in resizing_scales:
                if not isinstance(scale, float):
                    raise ValueError("Invalid resizing_scales, required a list of floats")
        except:
            raise ValueError("Invalid resizing_scales, required a list of floats")
        tiles = defaultdict(list)
        for img in image:
            new_img = torch_tensor_to_cv2_image(img)
            new_img = new_img.astype('float')
            height, width, channels = new_img.shape
            center = (width / 2, height / 2)
            for rotation in rotations:
                        rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1)
                        abs_cos = abs(rotation_matrix[0, 0])
                        abs_sin = abs(rotation_matrix[0, 1])
                        new_w = int(height * abs_sin + width * abs_cos)
                        new_h = int(height * abs_cos + width * abs_sin)
                        rotation_matrix[0, 2] += new_w / 2 - center[0]
                        rotation_matrix[1, 2] += new_h / 2 - center[1]

                        rotated_img = cv2.warpAffine(new_img, rotation_matrix, (new_w, new_h))
                        mode, rel_freq = mode_color(rotated_img, ignore_alpha=(channels == 4))

                        if mode is not None:
                            for scale in resizing_scales:
                                resized_tile = resize_image(rotated_img, scale)
                                res = tuple(resized_tile.shape[:2])

                                tiles[res].append({
                                    'tile': resized_tile,
                                    'mode': mode,
                                    'rel_freq': rel_freq
                                })
        return (tiles,)
    
    
    
    

class PC_TEST:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                      "mask": ("MASK",),
                    }
                }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tile",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "test"
    CATEGORY =  "😱 PointAgiClub"
    def test(self,**kwargs):
        print(kwargs)
        return "test"