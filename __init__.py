
from .tiler_comfy import TilerImage,TilerSelect,TileMaker,ImageListTileMaker,PC_TEST


NODE_CLASS_MAPPINGS = {

    "PC TilerImage": TilerImage,
    "PC TilerSelect":TilerSelect,
    "PC TileMaker":TileMaker,
    "PC ImageListTileMaker":ImageListTileMaker,
    "PC TEST":PC_TEST,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PC TilerImage": "TilerImage",
    "PC TilerSelect":"TilerSelect",
    "PC TileMaker":"TileMaker",
    "PC ImageListTileMaker":"ImageListTileMaker",
    "PC TEST":"PC_TEST",
}

WEB_DIRECTORY='./web'
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]