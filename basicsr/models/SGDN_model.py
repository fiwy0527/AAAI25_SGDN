from collections import OrderedDict
from os import path as osp
import kornia
import torch
from torch.nn import functional as F
from tqdm import tqdm
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class SGDNModel(SRModel):

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output, self.output_128, self.output_64 = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output, self.output_128, self.output_64 = self.net_g(self.lq)
            self.net_g.train()

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.gt_128 = F.interpolate(self.gt, scale_factor=0.5, mode='bilinear')
        self.gt_64 = F.interpolate(self.gt, scale_factor=0.25, mode='bilinear')
        self.output, self.output_128, self.output_64 = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        # pixel loss
        if self.cri_pix:
            l_pix1 = self.cri_pix(self.output, self.gt)
            l_pix2 = self.cri_pix(self.output_128, self.gt_128)
            l_pix3 = self.cri_pix(self.output_64, self.gt_64)
            pix_total = l_pix1 + l_pix2 + l_pix3
            l_total += pix_total
            loss_dict['pix_total'] = pix_total

        if self.cri_fft:
            l_fft256 = self.cri_fft(self.output, self.gt)
            l_fft128 = self.cri_fft(self.output_128, self.gt_128)
            l_fft64 = self.cri_fft(self.output_64, self.gt_64)
            l_fft_total = l_fft256 + l_fft128 + l_fft64
            l_total += l_fft_total
            loss_dict["l_fft_total"] = l_fft_total

        # perceptual loss
        if self.ssim_opt:
            ssim_loss1 = self.ssim_opt(self.output, self.gt)
            ssim_loss2 = self.ssim_opt(self.output_128, self.gt_128)
            ssim_loss3 = self.ssim_opt(self.output_64, self.gt_64)
            l_ssim_total = ssim_loss1 + ssim_loss2 + ssim_loss3
            l_total += l_ssim_total
            loss_dict['l_ssim_total'] = l_ssim_total


        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


