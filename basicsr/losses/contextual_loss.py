import sys

import torch
import torch.nn as nn
from .VGG_Model import VGG_Model
import torch.nn.functional as F
import copy
from basicsr.utils.registry import LOSS_REGISTRY


class Distance_Type:
    L2_Distance = 0
    L1_Distance = 1
    Cosine_Distance = 2

"""
config file is a dict.
layers_weights: dict, e.g., {'conv_1_1': 1.0, 'conv_3_2': 1.0}
crop_quarter: boolean

"""
layers = {
            "conv_1_1": 1.0,
            "conv_3_2": 1.0
        }

@LOSS_REGISTRY.register()
class Contextual_Loss(nn.Module):
    def __init__(self, layers_weights=layers, crop_quarter=False, max_1d_size=100, distance_type=Distance_Type.Cosine_Distance, b=1.0, h=0.1, cuda=True, loss_weight=1.0):
        super(Contextual_Loss, self).__init__()
        listen_list = []
        self.layers_weights = {}
        self.loss_weight = loss_weight
        try:
            listen_list = layers_weights.keys()
            self.layers_weights = layers_weights
        except:
            pass
        self.vgg_pred = VGG_Model(listen_list=listen_list)
        # self.vgg_gt = VGG_Model(listen_list=listen_list)
        # if cuda:
        #     self.vgg_pred = nn.DataParallel(self.vgg_pred.cuda())
            # self.vgg_gt = nn.DataParallel(self.vgg_gt.cuda())
        self.crop_quarter = crop_quarter
        self.distanceType = distance_type
        self.max_1d_size = max_1d_size
        self.b = b
        self.h = h


    def forward(self, images, gt):
        if images.device.type == 'cpu':
            loss = torch.zeros(1)
            vgg_images = self.vgg_pred(images)
            vgg_images = {k: v.clone() for k, v in vgg_images.items()}
            vgg_gt = self.vgg_pred(gt)
        else:
            id_cuda = torch.cuda.current_device()
            loss = 0
            vgg_images = self.vgg_pred(images)
            vgg_images = {k: v.clone().cuda(id_cuda) for k, v in vgg_images.items()}
            vgg_gt = self.vgg_pred(gt)
            vgg_gt = {k: v.cuda(id_cuda) for k, v in vgg_gt.items()}
        # print('images', [v.device for k, v in vgg_images.items()])
        # print('gt', [v.device for k, v in vgg_gt.items()])

        for key in self.layers_weights.keys():
            N, C, H, W = vgg_images[key].size()

            if self.crop_quarter:
                vgg_images[key] = self._crop_quarters()

            if H*W > self.max_1d_size**2:
                vgg_images[key] = self._random_pooling(vgg_images[key], output_1d_size=self.max_1d_size)
                vgg_gt[key] = self._random_pooling(vgg_gt[key], output_1d_size=self.max_1d_size)

            loss_t = self.calculate_CX_Loss(vgg_images[key], vgg_gt[key])
            # print(loss_t)
            loss += loss_t * self.layers_weights[key]
            # del vgg_images[key], vgg_gt[key]
        return loss * self.loss_weight


    @staticmethod
    def _random_sampling(tensor, n, indices):
        N, C, H, W = tensor.size()
        S = H * W
        tensor = tensor.view(N, C, S)
        if indices is None:
            indices = torch.randperm(S)[:n].contiguous().type_as(tensor).long()
            indices = indices.view(1, 1, -1).expand(N, C, -1)
        indices = Contextual_Loss._move_to_current_device(indices)

        # print('current_device', torch.cuda.current_device(), tensor.device, indices.device)
        res = torch.gather(tensor, index=indices, dim=-1)
        return res, indices

    @staticmethod
    def _move_to_current_device(tensor):
        if tensor.device.type == 'cuda':
            id = torch.cuda.current_device()
            return tensor.cuda(id)
        return tensor

    @staticmethod
    def _random_pooling(feats, output_1d_size=100):
        single_input = type(feats) is torch.Tensor

        if single_input:
            feats = [feats]

        N, C, H, W = feats[0].size()
        feats_sample, indices = Contextual_Loss._random_sampling(feats[0], output_1d_size**2, None)
        res = [feats_sample]

        for i in range(1, len(feats)):
            feats_sample, _ = Contextual_Loss._random_sampling(feats[i], -1, indices)
            res.append(feats_sample)

        res = [feats_sample.view(N, C, output_1d_size, output_1d_size) for feats_sample in res]

        if single_input:
            return res[0]
        return res

    @staticmethod
    def _crop_quarters(feature):
        N, fC, fH, fW = feature.size()
        quarters_list = []
        quarters_list.append(feature[..., 0:round(fH / 2), 0:round(fW / 2)])
        quarters_list.append(feature[..., 0:round(fH / 2), round(fW / 2):])
        quarters_list.append(feature[..., round(fH / 2), 0:round(fW / 2)])
        quarters_list.append(feature[..., round(fH / 2):, round(fW / 2):])

        feature_tensor = torch.cat(quarters_list, dim=0)
        return feature_tensor

    @staticmethod
    def _create_using_L2(I_features, T_features):
        """
        Calculating the distance between each feature of I and T
        :param I_features:
        :param T_features:
        :return: raw_distance: [N, C, H, W, H*W], each element of which is the distance between I and T at each position
        """
        assert I_features.size() == T_features.size()
        N, C, H, W = I_features.size()

        Ivecs = I_features.view(N, C, -1)
        Tvecs = T_features.view(N, C, -1)
        #
        square_I = torch.sum(Ivecs*Ivecs, dim=1, keepdim=False)
        square_T = torch.sum(Tvecs*Tvecs, dim=1, keepdim=False)
        # raw_distance
        raw_distance = []
        for i in range(N):
            Ivec, Tvec, s_I, s_T = Ivecs[i, ...], Tvecs[i, ...], square_I[i, ...], square_T[i, ...]
            # matrix multiplication
            AB = Ivec.permute(1, 0) @ Tvec
            dist = s_I.view(-1, 1) + s_T.view(1, -1) - 2*AB

            raw_distance.append(dist.view(1, H, W, H*W))
        raw_distance = torch.cat(raw_distance, dim=0)
        raw_distance = torch.clamp(raw_distance, 0.0)
        return raw_distance

    @staticmethod
    def _create_using_L1(I_features, T_features):
        assert I_features.size() == T_features.size()
        N, C, H, W = I_features.size()

        Ivecs = I_features.view(N, C, -1)
        Tvecs = T_features.view(N, C, -1)

        raw_distance = []
        for i in range(N):
            Ivec, Tvec = Ivecs[i, ...], Tvecs[i, ...]
            dist = torch.sum(
                torch.abs(Ivec.view(C, -1, 1) - Tvec.view(C, 1, -1)), dim=0, keepdim=False
            )
            raw_distance.append(dist.view(1, H, W, H*W))
        raw_distance = torch.cat(raw_distance, dim=0)
        return raw_distance

    @staticmethod
    def _centered_by_T(I, T):
        mean_T = T.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        # print(I.device, T.device, mean_T.device)
        return I-mean_T, T-mean_T

    @staticmethod
    def _normalized_L2_channelwise(tensor):
        norms = tensor.norm(p=2, dim=1, keepdim=True)
        return tensor / norms

    @staticmethod
    def _create_using_dotP(I_features, T_features):
        assert I_features.size() == T_features.size()
        I_features, T_features = Contextual_Loss._centered_by_T(I_features, T_features)
        I_features = Contextual_Loss._normalized_L2_channelwise(I_features)
        T_features = Contextual_Loss._normalized_L2_channelwise(T_features)

        N, C, H, W = I_features.size()
        cosine_dist = []
        for i in range(N):
            T_features_i = T_features[i].view(1, 1, C, H*W).permute(3, 2, 0, 1).contiguous()
            I_features_i = I_features[i].unsqueeze(0)
            dist = F.conv2d(I_features_i, T_features_i).permute(0, 2, 3, 1).contiguous()
            cosine_dist.append(dist)
        cosine_dist = torch.cat(cosine_dist, dim=0)
        cosine_dist = (1 - cosine_dist) / 2
        cosine_dist = cosine_dist.clamp(min=0.0)
        return cosine_dist


    @staticmethod
    def _calculate_relative_distance(raw_distance, epsilon=1e-5):
        """
        Normalizing the distances first as Eq. (2) in paper
        :param raw_distance:
        :param epsilon:
        :return:
        """
        div = torch.min(raw_distance, dim=-1, keepdim=True)[0]
        relative_dist = raw_distance / (div + epsilon)
        return relative_dist

    def calculate_CX_Loss(self, I_features, T_features):
        I_features = Contextual_Loss._move_to_current_device(I_features)
        T_features = Contextual_Loss._move_to_current_device(T_features)

        if torch.sum(torch.isnan(I_features)) == torch.numel(I_features) or torch.sum(torch.isinf(I_features)) == torch.numel(I_features):
            print(I_features)
            raise ValueError('NaN or Inf in I_features')
        if torch.sum(torch.isnan(T_features)) == torch.numel(T_features) or torch.sum(
                torch.isinf(T_features)) == torch.numel(T_features):
            print(T_features)
            raise ValueError('NaN or Inf in T_features')

        if self.distanceType == Distance_Type.L1_Distance:
            raw_distance = Contextual_Loss._create_using_L1(I_features, T_features)
        elif self.distanceType == Distance_Type.L2_Distance:
            raw_distance = Contextual_Loss._create_using_L2(I_features, T_features)
        else:
            raw_distance = Contextual_Loss._create_using_dotP(I_features, T_features)
        if torch.sum(torch.isnan(raw_distance)) == torch.numel(raw_distance) or torch.sum(
                torch.isinf(raw_distance)) == torch.numel(raw_distance):
            print(raw_distance)
            raise ValueError('NaN or Inf in raw_distance')

        relative_distance = Contextual_Loss._calculate_relative_distance(raw_distance)
        if torch.sum(torch.isnan(relative_distance)) == torch.numel(relative_distance) or torch.sum(
                torch.isinf(relative_distance)) == torch.numel(relative_distance):
            print(relative_distance)
            raise ValueError('NaN or Inf in relative_distance')
        del raw_distance

        exp_distance = torch.exp((self.b - relative_distance) / self.h)
        if torch.sum(torch.isnan(exp_distance)) == torch.numel(exp_distance) or torch.sum(
                torch.isinf(exp_distance)) == torch.numel(exp_distance):
            print(exp_distance)
            raise ValueError('NaN or Inf in exp_distance')
        del relative_distance
        # Similarity
        contextual_sim = exp_distance / torch.sum(exp_distance, dim=-1, keepdim=True)
        if torch.sum(torch.isnan(contextual_sim)) == torch.numel(contextual_sim) or torch.sum(
                torch.isinf(contextual_sim)) == torch.numel(contextual_sim):
            print(contextual_sim)
            raise ValueError('NaN or Inf in contextual_sim')
        del exp_distance
        max_gt_sim = torch.max(torch.max(contextual_sim, dim=1)[0], dim=1)[0]
        del contextual_sim
        CS = torch.mean(max_gt_sim, dim=1)
        CX_loss = torch.mean(-torch.log(CS))
        if torch.isnan(CX_loss):
            raise ValueError('NaN in computing CX_loss')
        return CX_loss



if __name__ == '__main__':
    from PIL import Image
    from torchvision.transforms import transforms
    import torch.nn.functional as F
    layers = {
            "conv_1_1": 1.0,
            "conv_3_2": 1.0
        }
    I = torch.rand(1, 3, 128, 128).cuda()
    T = torch.randn(1, 3, 128, 128).cuda()
    contex_loss = Contextual_Loss(layers, max_1d_size=64).cuda()
    print(contex_loss(I, T))