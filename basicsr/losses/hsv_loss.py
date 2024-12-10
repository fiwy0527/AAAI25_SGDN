import torch
from torch import nn as nn
from skimage.color import rgb2hsv
from basicsr.utils.registry import LOSS_REGISTRY
import kornia

@LOSS_REGISTRY.register()
class HSVLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(HSVLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.MSELoss()

    def np_to_torch(img_np):
      """
      Converts image in numpy.array to torch.Tensor.

      From C x W x H [0..1] to  C x W x H [0..1]

      :param img_np:
      :return:
      """
      return torch.from_numpy(img_np)[None, :]


    def torch_to_np(img_var):
      """
      Converts an image in torch.Tensor format to np.array.

      From 1 x C x W x H [0..1] to  C x W x H [0..1]
      :param img_var:
      :return:
      """
      return img_var.detach().cpu().numpy()[0]

    def forward(self, pred, target):
        # hsv = self.np_to_torch(rgb2hsv(self.torch_to_np(pred).transpose(1, 2, 0)))
        hsv = kornia.color.rgb_to_hsv(pred)
        cap_prior = hsv[:, :, :, 2] - hsv[:, :, :, 1]
        cap_loss =  self.criterion(cap_prior, torch.zeros_like(cap_prior))

        return self.loss_weight * cap_loss


