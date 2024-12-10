import torch
import torch.nn as nn
from torchvision.models import vgg19, vgg16
from collections import OrderedDict

# VGG 19
# vgg_layer = {
#     'conv_1_1': 0, 'conv_1_2': 2, 'pool_1': 4, 'conv_2_1': 5, 'conv_2_2': 7, 'pool_2': 9, 'conv_3_1': 10, 'conv_3_2': 12, 'conv_3_3': 14, 'conv_3_4': 16, 'pool_3': 18, 'conv_4_1': 19, 'conv_4_2': 21, 'conv_4_3': 23, 'conv_4_4': 25, 'pool_4': 27, 'conv_5_1': 28, 'conv_5_2': 30, 'conv_5_3': 32, 'conv_5_4': 34, 'pool_5': 36
# }
#
# vgg_layer_inv = {
#     0: 'conv_1_1', 2: 'conv_1_2', 4: 'pool_1', 5: 'conv_2_1', 7: 'conv_2_2', 9: 'pool_2', 10: 'conv_3_1', 12: 'conv_3_2', 14: 'conv_3_3', 16: 'conv_3_4', 18: 'pool_3', 19: 'conv_4_1', 21: 'conv_4_2', 23: 'conv_4_3', 25: 'conv_4_4', 27: 'pool_4', 28: 'conv_5_1', 30: 'conv_5_2', 32: 'conv_5_3', 34: 'conv_5_4', 36: 'pool_5'
# }
# VGG 16
vgg_layer = {
    'conv_1_1': 0, 'conv_1_2': 2, 'pool_1': 4, 'conv_2_1': 5, 'conv_2_2': 7, 'pool_2': 9, 'conv_3_1': 10, 'conv_3_2': 12, 'conv_3_3': 14, 'pool_3': 16, 'conv_4_1': 17, 'conv_4_2': 19, 'conv_4_3': 21, 'pool_4': 23, 'conv_5_1': 24, 'conv_5_2': 26, 'conv_5_3': 28, 'pool_5': 30
}

vgg_layer_inv = {
    0: 'conv_1_1', 2: 'conv_1_2', 4: 'pool_1', 5: 'conv_2_1', 7: 'conv_2_2', 9: 'pool_2', 10: 'conv_3_1', 12: 'conv_3_2', 14: 'conv_3_3', 16: 'pool_3', 17: 'conv_4_1', 19: 'conv_4_2', 21: 'conv_4_3', 23: 'pool_4', 24: 'conv_5_1', 26: 'conv_5_2', 28: 'conv_5_3', 30: 'pool_5'
}

class VGG_Model(nn.Module):
    def __init__(self, listen_list=None):
        super(VGG_Model, self).__init__()
        vgg = vgg16(pretrained=True)
        self.vgg_model = vgg.features
        vgg_dict = vgg.state_dict()
        vgg_f_dict = self.vgg_model.state_dict()
        vgg_dict = {k: v for k, v in vgg_dict.items() if k in vgg_f_dict}
        vgg_f_dict.update(vgg_dict)
        # no grad
        for p in self.vgg_model.parameters():
            p.requires_grad = False
        if listen_list == []:
            self.listen = []
        else:
            self.listen = set()
            for layer in listen_list:
                self.listen.add(vgg_layer[layer])
        self.features = OrderedDict()

    def forward(self, x):
        for index, layer in enumerate(self.vgg_model):
            x = layer(x)
            if index in self.listen:
                self.features[vgg_layer_inv[index]] = x
        return self.features