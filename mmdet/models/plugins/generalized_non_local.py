import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from mmcv.cnn import kaiming_init


class GeneralizedNonLocal2D(nn.Module):

    def __init__(self,
                 in_dim,
                 num_mixtures=9,
                 position_embedding_dim=-1,
                 position_magnitude=1,
                 kv_stride=2,
                 q_stride=1):

        super(GeneralizedNonLocal2D, self).__init__()

        self.position_embedding_dim = (
            position_embedding_dim if position_embedding_dim > 0 else in_dim)

        self.position_magnitude = position_magnitude
        self.num_mixtures = num_mixtures
        self.channel_in = in_dim
        self.kv_stride = kv_stride
        self.q_stride = q_stride
        self.qk_embed_dim = in_dim // num_mixtures
        out_c = self.qk_embed_dim * num_mixtures
        
        # query (content) - key (content)
        self.query_conv = nn.Conv2d(
                in_channels=in_dim,
                out_channels=out_c,
                kernel_size=1,
                bias=False)
        self.query_conv.kaiming_init = True
        
        self.key_conv = nn.Conv2d(
                in_channels=in_dim,
                out_channels=out_c,
                kernel_size=1,
                bias=False)
        self.key_conv.kaiming_init = True
        
        self.v_dim = in_dim // num_mixtures
        self.value_conv = nn.Conv2d(
                in_channels=in_dim,
                out_channels=self.v_dim * num_mixtures,
                kernel_size=1,
                bias=False)
        self.value_conv.kaiming_init = True

        # query (content) - key (relative position)
        self.appr_geom_fc_x = nn.Linear(
            self.position_embedding_dim // 2, out_c, bias=False)
        self.appr_geom_fc_x.kaiming_init = True
        
        self.appr_geom_fc_y = nn.Linear(
            self.position_embedding_dim // 2, out_c, bias=False)
        self.appr_geom_fc_y.kaiming_init = True
        
        self.proj_conv = nn.Conv2d(
            in_channels=self.v_dim * num_mixtures,
            out_channels=in_dim,
            kernel_size=1,
            bias=True)
        self.proj_conv.kaiming_init = True
        self.gamma = nn.Parameter(torch.zeros(1))
        self.context_conv = nn.Conv1d(
            in_channels=self.qk_embed_dim * num_mixtures,
            out_channels=num_mixtures,
            kernel_size=1)
        if self.q_stride > 1:
            self.q_downsample = nn.AvgPool2d(
                kernel_size=1, stride=self.q_stride)
        else:
            self.q_downsample = None

        if self.kv_stride > 1:
            self.kv_downsample = nn.AvgPool2d(
                kernel_size=1, stride=self.kv_stride)
        else:
            self.kv_downsample = None

        self.init_weights()

    def get_position_embedding(self,
                               h,
                               w,
                               h_kv,
                               w_kv,
                               q_stride,
                               kv_stride,
                               device,
                               feat_dim,
                               wave_length=1000):
        h_idxs = torch.linspace(0, h - 1, h).cuda(device)
        h_idxs = h_idxs.view((h, 1)) * q_stride

        w_idxs = torch.linspace(0, w - 1, w).cuda(device)
        w_idxs = w_idxs.view((w, 1)) * q_stride

        h_kv_idxs = torch.linspace(0, h_kv - 1, h_kv).cuda(device)
        h_kv_idxs = h_kv_idxs.view((h_kv, 1)) * kv_stride

        w_kv_idxs = torch.linspace(0, w_kv - 1, w_kv).cuda(device)
        w_kv_idxs = w_kv_idxs.view((w_kv, 1)) * kv_stride

        # (h, h_kv, 1)
        h_diff = h_idxs.unsqueeze(1) - h_kv_idxs.unsqueeze(0)
        h_diff *= self.position_magnitude

        # (w, w_kv, 1)
        w_diff = w_idxs.unsqueeze(1) - w_kv_idxs.unsqueeze(0)
        w_diff *= self.position_magnitude

        feat_range = torch.arange(0, feat_dim / 4).cuda(device)

        dim_mat = torch.Tensor([wave_length]).cuda(device)
        dim_mat = dim_mat**((4. / feat_dim) * feat_range)
        dim_mat = dim_mat.view((1, 1, -1))

        embedding_x = torch.cat(
            ((w_diff / dim_mat).sin(), (w_diff / dim_mat).cos()), dim=2)

        embedding_y = torch.cat(
            ((h_diff / dim_mat).sin(), (h_diff / dim_mat).cos()), dim=2)

        return embedding_x, embedding_y

    def forward(self, x_input):
        num_mixtures = self.num_mixtures

        # use empirical_attention
        if self.q_downsample is not None:
            x_q = self.q_downsample(x_input)
        else:
            x_q = x_input
        n, _, h, w = x_q.shape

        if self.kv_downsample is not None:
            x_kv = self.kv_downsample(x_input)
        else:
            x_kv = x_input
        _, _, h_kv, w_kv = x_kv.shape
        
        # query projection
        proj_query = self.query_conv(x_q).view(
            (n, num_mixtures, self.qk_embed_dim, h * w))
        proj_query = proj_query.permute(0, 1, 3, 2)
    
        # key projection
        proj_key = self.key_conv(x_kv).view(
            (n, num_mixtures, self.qk_embed_dim, h_kv * w_kv))
        
        # postion x,y embedding
        position_embed_x, position_embed_y = self.get_position_embedding(
            h, w, h_kv, w_kv, self.q_stride, self.kv_stride,
            x_input.device, self.position_embedding_dim)
        
#         positon x embedding projection
#         (n, num_mixtures, w, w_kv, dim)
        position_feat_x = self.appr_geom_fc_x(position_embed_x).\
            view(1, w, w_kv, num_mixtures, self.qk_embed_dim).\
            permute(0, 3, 1, 2, 4).\
            repeat(n, 1, 1, 1, 1)
        
#         position y embedding projection
#         (n, num_mixtures, h, h_kv, dim)
        position_feat_y = self.appr_geom_fc_y(position_embed_y).\
            view(1, h, h_kv, num_mixtures, self.qk_embed_dim).\
            permute(0, 3, 1, 2, 4).\
            repeat(n, 1, 1, 1, 1)

        position_feat_x /= math.sqrt(2)
        position_feat_y /= math.sqrt(2)
        
        # content - content energy
        energy = torch.matmul(proj_query, proj_key).\
            view(n, num_mixtures, h, w, h_kv, w_kv)
        
        # content - relation_position energy
        proj_query_reshape = proj_query.\
            view(n, num_mixtures, h, w, self.qk_embed_dim)
        proj_query_reshape_y = proj_query_reshape
        proj_query_reshape_x = proj_query_reshape.\
            permute(0, 1, 3, 2, 4)
        
        position_feat_x_reshape = position_feat_x.\
            permute(0, 1, 2, 4, 3)
        position_feat_y_reshape = position_feat_y.\
            permute(0, 1, 2, 4, 3)

        energy_x = torch.matmul(proj_query_reshape_x,
                                position_feat_x_reshape)
        energy_x = energy_x.permute(0, 1, 3, 2, 4).unsqueeze(4)
                    
        energy_y = torch.matmul(proj_query_reshape_y,
                                position_feat_y_reshape)            
        energy_y = energy_y.unsqueeze(5)

        energy += energy_x + energy_y
                    
        energy = energy.view(n, num_mixtures, h * w, h_kv * w_kv)

        attention = F.softmax(energy, 3)
        
        context = proj_query.\
                permute(0,1,3,2).\
                contiguous().\
                view(n, num_mixtures * self.qk_embed_dim, -1)
        context = torch.mean(context, dim=2, keepdim=True)
        mixture_weight = F.softmax(self.context_conv(context), dim=1)
        attention = torch.sum(attention * mixture_weight.unsqueeze(2), dim=1)
        
        proj_value = self.value_conv(x_kv)
        proj_value_reshape = proj_value.\
            view((n, self.v_dim * self.num_mixtures, h_kv * w_kv)).\
            permute(0, 2, 1)

        out = torch.matmul(attention, proj_value_reshape).\
            permute(0, 2, 1).\
            contiguous().\
            view(n, self.v_dim * self.num_mixtures, h, w)
        
        out = self.proj_conv(out)
        out = self.gamma * out + x_input
        return out

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'kaiming_init') and m.kaiming_init:
                kaiming_init(
                    m,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                    bias=0,
                    distribution='uniform',
                    a=1)
