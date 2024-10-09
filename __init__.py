
from .tiler_comfy import TilerImage,TilerSelect


NODE_CLASS_MAPPINGS = {

    "PC TilerImage": TilerImage,
    "PC TilerSelect":TilerSelect,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PC TilerImage": "TilerImage",
    "PC TilerSelect":"TilerSelect",
}

WEB_DIRECTORY='./web'
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]