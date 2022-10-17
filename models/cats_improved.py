from functools import reduce
from operator import add
from contextlib import nullcontext

import torch
import torch.nn as nn
from torchvision.models import resnet
from torchvision.models import vgg

try:
    import clip
except:
    print('CLIP is not installed! "pip install git+https://github.com/openai/CLIP.git" to use CLIP backbone.')

from .base.feature import extract_feat_clip, extract_feat_vgg, extract_feat_res
from .base.correlation import Correlation
from .cats_model import CATs


class CATsImproved(nn.Module):
    def __init__(self, backbone='resnet101', freeze=False):
        super().__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        elif backbone == 'clip_resnet101':
            self.backbone = clip.load("RN101")[0].float()
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_clip
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()
        self.hpn_learner = CATs(inch=list(reversed(nbottlenecks[-3:])))
        self.freeze = freeze

    def stack_feats(self, feats):

        feats_l4 = torch.stack(feats[-self.stack_ids[0]:]).transpose(0, 1)
        feats_l3 = torch.stack(feats[-self.stack_ids[1]:-self.stack_ids[0]]).transpose(0, 1)
        feats_l2 = torch.stack(feats[-self.stack_ids[2]:-self.stack_ids[1]]).transpose(0, 1)

        return [feats_l4, feats_l3, feats_l2]

    def forward(self, trg_img, src_img):
        with torch.no_grad() if self.freeze else nullcontext():
            trg_feats = self.extract_feats(trg_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            src_feats = self.extract_feats(src_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)

        corr = Correlation.multilayer_correlation(trg_feats, src_feats, self.stack_ids)

        flow = self.hpn_learner(corr, self.stack_feats(trg_feats), self.stack_feats(src_feats))

        return flow
