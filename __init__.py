from .tiler_comfy import TilerImage, DefaultTilerSelect, TileMaker, ImageListTileMaker


NODE_CLASS_MAPPINGS = {
    "PC TilerImage": TilerImage,
    "PC DefaultTilerSelect": DefaultTilerSelect,
    "PC TileMaker": TileMaker,
    "PC ImageListTileMaker": ImageListTileMaker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PC TilerImage": "TilerImage",
    "PC DefaultTilerSelect": "DefaultTilerSelect",
    "PC TileMaker": "TileMaker",
    "PC ImageListTileMaker": "ImageListTileMaker",
}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
