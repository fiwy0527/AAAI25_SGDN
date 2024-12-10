import torch
from torch import nn as nn
from torch.nn import functional as F
from pytorch_msssim import SSIM, MS_SSIM
from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss



@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):
    def __init__(self, channels, loss_weight=0.1):
        super(SSIMLoss, self).__init__()
        self.loss_weight = loss_weight
        self.ssim = SSIM(data_range=1., size_average=True, channel=channels)

    def forward(self, output, target):
        ssim_loss = (1 - self.ssim(output, target)) * self.loss_weight
        return ssim_loss