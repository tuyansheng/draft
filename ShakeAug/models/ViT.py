import torch
from timm.models.vision_transformer import vit_small_patch16_224 as timm_vit_small_patch16_224
from timm.models.vision_transformer import vit_tiny_patch16_224 as timm_vit_tiny_patch16_224
from timm.models.vision_transformer import vit_small_patch32_224 as timm_vit_small_patch32_224
from timm.models.vision_transformer import vit_base_patch16_224 as timm_vit_base_patch16_224
from timm.models.vision_transformer import vit_base_patch32_224 as timm_vit_base_patch32_224

def vit_tiny_patch16_224(num_classes: int, **kwargs):
    model = timm_vit_tiny_patch16_224(pretrained=False, num_classes=num_classes)
    return model

def vit_small_patch16_224(num_classes: int, **kwargs):
    model = timm_vit_small_patch16_224(pretrained=False, num_classes=num_classes)
    return model

def vit_small_patch32_224(num_classes: int, **kwargs):
    model = timm_vit_small_patch32_224(pretrained=False, num_classes=num_classes)
    return model

def vit_base_patch16_224(num_classes: int, **kwargs):
    model = timm_vit_base_patch16_224(pretrained=False, num_classes=num_classes)
    return model

def vit_base_patch32_224(num_classes: int, **kwargs):
    model = timm_vit_base_patch32_224(pretrained=False, num_classes=num_classes)
    return model
