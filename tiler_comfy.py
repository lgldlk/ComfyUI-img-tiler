import cv2
import os
from collections import defaultdict
from .utils import (
    torch_tensor_to_cv2_image,
    color_quantization,
    cv2_image_to_torch_tensor,
    get_all_directories,
    merge_tiles,
    get_processed_image_boxes,
    create_tiled_image,
    load_tiles,
    mode_color,
    resize_image,
    find_most_common_scale,
    euclidean_distance,
    resize_and_crop,
    composite,
    add_alpha_channel,
    find_files_with_extensions,
)
import re
import json
import torch
import numpy as np
from torchvision import transforms
from PIL import Image


now_folder_path = os.path.dirname(__file__)

tilesList = []
for dir in get_all_directories(os.path.join(now_folder_path, "tiles")):
    dir_res = re.search(r"[\\|\/]gen_(.*)", dir)
    if dir_res:
        tilesList.append({"name": dir_res.group(1), "path": dir, "type": "tile"})

for dir in find_files_with_extensions(
    os.path.join(now_folder_path, "tiles"), "png", ["gen_"]
):
    dir_res = re.search(r"[\\|\/](.*).png", dir)
    if dir_res:
        tilesList.append({"name": dir_res.group(1), "path": dir, "type": "image"})

default_resizing_scales = "[0.6, 0.3, 0.15]"


class DefaultTilerSelect:
    @classmethod
    def INPUT_TYPES(cls):
        select_tile_options = []
        for tile in tilesList:
            select_tile_options.append(tile["name"])
        return {
            "required": {
                "tile": (select_tile_options, {"default": "square_50"}),
            }
        }

    RETURN_TYPES = ("Pc_Tiles",)
    RETURN_NAMES = ("tile",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "run_tile"
    CATEGORY = "😱 PointAgiClub"

    def run_tile(self, tile):
        result = next((x for x in tilesList if x["name"] == tile), None)
        if result[type] == "image":
            img = Image.open(result["path"])
            transform = transforms.ToTensor()
            result["tile"] = transform(img).unsqueeze(0)
        return ([result],)


class TilerImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth": ("INT", {"default": 6, "min": 2, "max": 16}),
                "overlap_tiles": ("BOOLEAN", {"default": False}),
                "auto_perfect_grid": ("BOOLEAN", {"default": True}),
                "size_num": ("INT", {"default": 3, "min": 1, "max": 6}),
                "max_tile_size": ("INT", {"default": 100, "min": 1, "max": 4096}),
                "tile1": ("Pc_Tiles",),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "create_tiled_image"
    CATEGORY = "😱 PointAgiClub"

    def create_tiled_image(
        self, image, overlap_tiles, auto_perfect_grid, mask=None, **kwargs
    ):
        tiles = defaultdict(list)
        for k, v in kwargs.items():
            tiles = merge_tiles(tiles, v)
        tile_keys = list(tiles.keys())

        print(f"Tiles loaded: {len(tile_keys)} sizes: {tile_keys}")
        if auto_perfect_grid:
            most_common_size, non_multiple_sizes = find_most_common_scale(tile_keys)
            if most_common_size is not None:
                for point_b in non_multiple_sizes:
                    closest_point = min(
                        [point_a for point_a in most_common_size if point_a < point_b],
                        key=lambda point_a: euclidean_distance(point_b, point_a),
                    )
                    print(f"Closest point to {point_b} is {closest_point}")
                    # 调整 tiles  point_b to closest_point
                    for tile in tiles[point_b]:
                        tile["tile"] = resize_and_crop(
                            tile["tile"], closest_point[1], closest_point[0]
                        )
                        mode, rel_freq = mode_color(tile["tile"], ignore_alpha=True)
                        tile["mode"] = mode
                        tile["rel_freq"] = rel_freq
                    tiles[closest_point] += tiles.pop(point_b)
        if mask is not None:
            old_img = image.clone()

        image = torch_tensor_to_cv2_image(image)
        boxes, original_res = get_processed_image_boxes(image, tiles)
        img = create_tiled_image(boxes, original_res, overlap_tiles)
        res = cv2_image_to_torch_tensor(img)
        if mask is not None:
            res = composite(
                add_alpha_channel(old_img).movedim(-1, 1),
                res.movedim(-1, 1),
                0,
                0,
                mask,
                1,
            ).movedim(1, -1)
        return (res,)


class TileMaker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "rotations": ("STRING", {"default": "[0]"}),
                "expand": ("BOOLEAN", {"default": True}),
                # "output_folder": ("STRING", {"default": "generated_images"}),
            }
        }

    RETURN_TYPES = ("Pc_Tiles",)
    RETURN_NAMES = ("tile",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "generate_images"
    CATEGORY = "😱 PointAgiClub"

    def generate_images(
        self,
        image: torch.Tensor,
        rotations,
        expand,
        # output_folder
    ):
        try:
            rotations = json.loads(rotations)
            if not all(isinstance(rotation, (int, float)) for rotation in rotations):
                raise ValueError(
                    "Invalid rotations format, should be a list of numbers"
                )
        except:
            raise ValueError(
                "Invalid rotations format, should be a JSON formatted list of numbers"
            )
        tiles = []
        for img in image:
            for rotation in rotations:
                if rotation == 0:
                    rotated_img = img
                else:
                    # Rotate the image tensor
                    rotated_img = transforms.functional.rotate(
                        img, angle=rotation, expand=expand
                    )
                tiles.append({"tile": rotated_img, "type": "image"})
        return (tiles,)


class ImageListTileMaker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "rotations": ("STRING", {"default": "[0]"}),
                "resizing_scales": ("STRING", {"default": default_resizing_scales}),
                # "output_folder": ("STRING", {"default": "generated_images"}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("Pc_Tiles",)
    RETURN_NAMES = ("tile",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "generate_images"
    CATEGORY = "😱 PointAgiClub"

    def generate_images(
        self,
        image,
        rotations,
        resizing_scales,
        # output_folder
    ):

        try:
            rotations = json.loads(rotations[0])
            if not all(isinstance(rotation, (int, float)) for rotation in rotations):
                raise ValueError(
                    "Invalid rotations format, should be a list of numbers"
                )
        except:
            raise ValueError(
                "Invalid rotations format, should be a JSON formatted list of numbers"
            )
        try:
            resizing_scales = json.loads(resizing_scales[0])
            for scale in resizing_scales:
                if not isinstance(scale, float):
                    raise ValueError(
                        "Invalid resizing_scales, required a list of floats"
                    )
        except:
            raise ValueError("Invalid resizing_scales, required a list of floats")
        tiles = defaultdict(list)
        for img in image:
            new_img = torch_tensor_to_cv2_image(img)
            new_img = new_img.astype("float")
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

                        tiles[res].append(
                            {"tile": resized_tile, "mode": mode, "rel_freq": rel_freq}
                        )
        return (tiles,)
