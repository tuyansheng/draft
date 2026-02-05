import os
import torch
from models.VGG import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn 
from models.Resnet import ResNet_18, ResNet_34, ResNet_50, ResNet_101, ResNet_152
from models.DenseNet import DenseNet121, DenseNet161, DenseNet169, DenseNet201
from models.ConNeXt import convnext_tiny, convnext_small,convnext_base,convnext_large,convnext_xlarge
from models.Swin import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, swin_base_patch4_window7_224, swin_base_patch4_window12_384
from models.PerViT import pervit_tiny, pervit_small, pervit_medium
from models.UniFormer import uniformer_small, uniformer_base, uniformer_base_ls
from models.CrossViT import crossvit_tiny_224, crossvit_small_224, crossvit_base_224
from models.ViT import vit_base_patch16_224, vit_tiny_patch16_224,vit_small_patch16_224


class Experiment_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'ResNet_18': ResNet_18,
            'ResNet_34': ResNet_34, 
            'ResNet_50': ResNet_50, 
            'ResNet_101': ResNet_101, 
            'ResNet_152': ResNet_152,
            'vgg11': vgg11,
            'vgg11_bn': vgg11_bn,
            'vgg13': vgg13,
            'vgg13_bn': vgg13_bn,
            'vgg16': vgg16, 
            'vgg16_bn': vgg16_bn, 
            'vgg19': vgg19, 
            'vgg19_bn': vgg19_bn,
            'DenseNet121': DenseNet121,
            'DenseNet161': DenseNet161, 
            'DenseNet169': DenseNet169, 
            'DenseNet201': DenseNet201,
            'ConvNeXt_T': convnext_tiny,
            'ConvNeXt_S': convnext_small,
            'ConvNeXt_B': convnext_base,
            'ConvNeXt_L': convnext_large,
            'ConvNeXt_XL': convnext_xlarge,
            'ViT_T': vit_tiny_patch16_224,
            'ViT_S': vit_small_patch16_224,
            'ViT_B': vit_base_patch16_224,
            'Swin_T': swin_tiny_patch4_window7_224,
            'Swin_S': swin_small_patch4_window7_224,
            'Swin_B': swin_base_patch4_window7_224,
            'Swin_B_384': swin_base_patch4_window12_384,
            'PerViT_T': pervit_tiny,
            'PerViT_S': pervit_small,
            'PerViT_M': pervit_medium,
            'UniFormer_S': uniformer_small,
            'UniFormer_B': uniformer_base,
            'UniFormer_BPS': uniformer_base_ls,
            'CrossViT_T': crossvit_tiny_224,
            'CrossViT_S': crossvit_small_224,
            'CrossViT_B': crossvit_base_224,
        }

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)


    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device =  torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
        return device
    
    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


