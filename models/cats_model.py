import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from .base.conv4d import Encoder4D, transpose4d, Interpolate4d
from .base.transformer import Transformer
from models.mod import unnormalise_and_convert_mapping_to_flow


class CATs(nn.Module):
    def __init__(self, inch, affinity_dropout=.5, nhead=4):
        super(CATs, self).__init__()
        self.final_corr_size = 32

        self.early_conv = nn.ModuleList([
            Encoder4D( # Encoder for conv_5
                corr_levels=(inch[0], nhead),
                kernel_size=(
                    (3, 3, 3, 3),
                ),
                stride=(
                    (2, 2, 2, 2),
                ),
                padding=(
                    (1, 1, 1, 1),
                ),
                group=(1,),
            ),
            Encoder4D( # Encoder for conv_4
                corr_levels=(inch[1], nhead),
                kernel_size=(
                    (3, 3, 3, 3),
                ),
                stride=(
                    (2, 2, 2, 2),
                ),
                padding=(
                    (1, 1, 1, 1),
                ),
                group=(1,),
            ),
            Encoder4D( # Encoder for conv_3
                corr_levels=(inch[2], nhead),
                kernel_size=(
                    (3, 3, 3, 3),
                ),
                stride=(
                    (2, 2, 2, 2),
                ),
                padding=(
                    (1, 1, 1, 1),
                ),
                group=(1,),
            ),
        ])
        
        self.transformers = nn.ModuleList([
            Transformer(
                in_channel=nhead,
                depth=1,
                affinity_dim=128,
                target_proj_dim=512,
                nhead=nhead,
                mlp_ratio=4.,
                input_corr_size=(8, 8, 8, 8),
                stride=(1, 1, 1, 1),
                group=1,
            ),
            Transformer(
                in_channel=nhead,
                depth=1,
                affinity_dim=128,
                target_proj_dim=512,
                nhead=nhead,
                mlp_ratio=4.,
                input_corr_size=(16, 16, 16, 16),
                stride=(2, 2, 1, 1),
                group=1,
            ),
            Transformer(
                in_channel=nhead,
                depth=1,
                affinity_dim=128,
                target_proj_dim=512,
                nhead=nhead,
                mlp_ratio=4.,
                input_corr_size=(32, 32, 32, 32),
                kernel_size=(5, 5, 3, 3),
                stride=(4, 4, 1, 1),
                padding=(2, 2, 1, 1),
                group=1,
            ),
        ])

        self.affinity_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(True),
                nn.Dropout2d(affinity_dropout)
            ),
            nn.Sequential(
                nn.Conv2d(1024, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(True),
                nn.Dropout2d(affinity_dropout)
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(True),
                nn.Dropout2d(affinity_dropout)
            )
        ])

        self.upscale = nn.ModuleList([
            Interpolate4d(size=(16, 16, 16, 16)),
            Interpolate4d(size=(32, 32, 32, 32)),
        ])
    
        self.x_normal = np.linspace(-1,1,self.final_corr_size)
        self.x_normal = nn.Parameter(torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False))
        self.y_normal = np.linspace(-1,1,self.final_corr_size)
        self.y_normal = nn.Parameter(torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False))
        
    def softmax_with_temperature(self, x, beta, d = 1):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M # subtract maximum value for stability
        exp_x = torch.exp(x/beta)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum

    def soft_argmax(self, corr, beta=0.02):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        b,_,h,w = corr.size()
        
        corr = self.softmax_with_temperature(corr, beta=beta, d=1)
        corr = corr.view(-1,h,w,h,w) # (target hxw) x (source hxw)

        grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
        x_normal = self.x_normal.expand(b,w)
        x_normal = x_normal.view(b,w,1,1)
        grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        
        grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
        y_normal = self.y_normal.expand(b,h)
        y_normal = y_normal.view(b,h,1,1)
        grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        return grid_x, grid_y

    def forward(self, hypercorr_pyramid, target_feats, source_feats):
        target_feats = [proj(x[:, -1]) for x, proj in zip(target_feats, self.affinity_proj)]
        source_feats = [proj(x[:, -1]) for x, proj in zip(source_feats, self.affinity_proj)]

        corr5 = self.early_conv[0](hypercorr_pyramid[0])
        corr4 = self.early_conv[1](hypercorr_pyramid[1])
        corr3 = self.early_conv[2](hypercorr_pyramid[2])

        corr5 = corr5 + self.transformers[0](corr5, source_feats[0]) + transpose4d(self.transformers[0](transpose4d(corr5), target_feats[0]))
        corr4 = corr4 + self.upscale[0](corr5)
        corr4 = corr4 + self.transformers[1](corr4, source_feats[1]) + transpose4d(self.transformers[1](transpose4d(corr4), target_feats[1]))
        corr3 = corr3 + self.upscale[1](corr4)
        corr3 = corr3 + self.transformers[2](corr3, source_feats[2]) + transpose4d(self.transformers[2](transpose4d(corr3), target_feats[2]))
        corr3 = corr3.mean(1)

        grid_x, grid_y = self.soft_argmax(rearrange(corr3, 'B H_t W_t H_s W_s -> B (H_s W_s) H_t W_t'))

        grid = torch.cat((grid_x, grid_y), dim=1)
        flow = unnormalise_and_convert_mapping_to_flow(grid)

        return flow