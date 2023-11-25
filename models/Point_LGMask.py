import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
from extensions.pointops.functions import pointops

from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
import numpy as np
from collections import OrderedDict
from torchvision import transforms
from datasets import data_transforms
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from .detr.build import build_encoder as build_encoder_3detr, build_preencoder as build_preencoder_3detr

train_transforms_1 = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslateAndJitter(),
    ]
)
train_transforms_2 = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslateAndJitter(),
    ]
)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


### ref https://github.com/Strawberry-Eat-Mango/PCT_Pytorch/blob/main/util.py ###
def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class Coss_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, kv, q):
        B, N, C = kv.shape
        _, N_m, _ = q.shape

        q = self.wq(q).reshape(B, N_m, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(kv).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(kv).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        z = (attn @ v).transpose(1, 2).reshape(B, N_m, C)
        z = self.proj(z)
        z = self.proj_drop(z)
        return z


class Coss_attention_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1_1 = norm_layer(dim)
        self.norm1_2 = norm_layer(dim)
        self.norm1_3 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Coss_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, kv, q):
        q = q + self.drop_path(self.attn(self.norm1_1(kv), self.norm1_2(q)))
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q


class Coss_attention_T_Encoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Coss_attention_Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, kv, q):
        for _, block in enumerate(self.blocks):
            q = block(kv, q)
        pre_mask_f = self.head(self.norm(q))
        return pre_mask_f


class Group_maskpoint_only_neig(nn.Module):
    def __init__(self, group_size):
        super().__init__()
        self.group_size = group_size

    def forward(self, xyz, center):
        '''
            input: B N 3
            center : B G 3
            ---------------------------
            output: B G M 3

        '''

        idx = pointops.knn(center, xyz, self.group_size)[0]
        neighborhood = pointops.index_points(xyz, idx)
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood


class Group_maskpoint(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        center = pointops.fps(xyz, self.num_group)
        idx = pointops.knn(center, xyz, self.group_size)[0]
        neighborhood = pointops.index_points(xyz, idx)
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class DummyGroup(Group_maskpoint):
    def forward(self, xyz):
        center = xyz
        neighborhood = torch.zeros_like(xyz).unsqueeze(2)
        return neighborhood, center


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        # self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, visFalg=0):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        idx = knn_point(self.group_size, xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        if visFalg == 1:
            return neighborhood, center, (neighborhood + center.unsqueeze(2))
        return neighborhood, center


class Crop_pc(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        # self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, R_min, R_max):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''

        # first downSample
        _, num_points, _ = xyz.shape
        assert R_max > R_min
        crop_rate = np.random.random() * (R_max - R_min) + R_min

        # reason: self.group_size / (num_points * downSample_rate) == crop_rate
        downSample_rate = self.group_size / (num_points * crop_rate)
        downSampel_pc_num = int(num_points * downSample_rate)
        downSampel_pc = misc.fps(xyz, downSampel_pc_num)

        xyz = downSampel_pc

        # go on  Group()
        batch_size, num_points, _ = xyz.shape

        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3

        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        idx = knn_point(self.group_size, xyz, center)  # B G M

        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        cls_x = []
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
            cls_x.append(x[:, 0])
        return x, cls_x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim

        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)


        self.cls_head = config.cls_head


        # for MN40
        if self.cls_head == 13:
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )
        # for scanObjectNN pb
        elif self.cls_head == 15:  # all cls  + all token
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * (self.depth + self.num_group), 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )
        # for scanObjectNN ob
        elif self.cls_head == 20:  # all cls  + all token, mlp 1024  start
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * (self.depth + self.num_group), 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )
        # for scanObjectNN oo
        elif self.cls_head == 50:  # all cls  + all token  384
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * (self.depth + self.num_group), 384),
                nn.BatchNorm1d(384),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(384, 384),
                nn.BatchNorm1d(384),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(384, self.cls_dim)
            )
        # for few-shot  on MN40 and scanObjectNN
        elif self.cls_head == 43:
            dim_val = 384
            self.lastChanlDim = nn.Sequential(
                nn.Linear(self.trans_dim, dim_val)
            )
            self.cls_head_finetune = nn.Sequential(

                nn.Linear(self.trans_dim * 3 + dim_val * self.num_group, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )



        self.build_loss_func()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
        # self.loss_ce_smooth = nn.CrossEntropyLoss(label_smoothing=1)

    def get_loss_acc(self, pred, gt):
        # import pdb; pdb.set_trace()
        gt = gt.contiguous().view(-1).long()

        loss = self.loss_ce(pred, gt.long())

        pred = pred.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))

        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('pc_msn_target_encoder') and not k.startswith('pc_msn_target_encoder.p_head'):
                base_ckpt[k[len('pc_msn_target_encoder.'):]] = base_ckpt[k]
                a = base_ckpt[k]
                b = base_ckpt[k[len('pc_msn_target_encoder.'):]]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger='Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger='Transformer'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger='Transformer'
            )

        print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')

    def forward(self, pts, tsneFlag=False):
        # divide the point clo  ud in the same form. This is important
        neighborhood, center = self.group_divider(pts)

        group_input_tokens = self.encoder(neighborhood)  # B G N

        batch_size, seq_len, C = group_input_tokens.size()

        # add pos embedding
        pos = self.pos_embed(center)

        # prepare cls and mask
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cls_pos = self.cls_pos.expand(batch_size, -1, -1)

        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        if tsneFlag == True:
            x, _ = self.blocks(x, pos)
            x = self.norm(x)

            concat_f = torch.cat((x[:, 0], x[:, 1:].max(1)[0], x[:, 1:].mean(1)), dim=-1)

            return concat_f


        if self.cls_head == 13:
            x, _ = self.blocks(x, pos)
            x = self.norm(x)
            concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
            ret = self.cls_head_finetune(concat_f)
            return ret
        elif self.cls_head == 15:  # all cls  + all token
            x, cls_x = self.blocks(x, pos)
            cls_x = torch.cat(cls_x, dim=-1)
            all_token = x[:, 1:].reshape(batch_size, -1)
            concat_f = torch.cat([cls_x, all_token], dim=-1)
            ret = self.cls_head_finetune(concat_f)
            return ret
        elif self.cls_head == 20:  # cls  + max(all token),mlp start  1024
            x, cls_x = self.blocks(x, pos)
            cls_x = torch.cat(cls_x, dim=-1)
            all_token = x[:, 1:].reshape(batch_size, -1)
            concat_f = torch.cat([cls_x, all_token], dim=-1)
            ret = self.cls_head_finetune(concat_f)
            return ret
        elif self.cls_head == 43:  # [3,7,11] + all_token dim 384
            x, cls_x = self.blocks(x, pos)
            x = self.norm(x)
            cls_t = [cls_x[3], cls_x[7], cls_x[11]]
            cls_t = torch.cat(cls_t, dim=-1)
            all_token = x[:, 1:].reshape(batch_size, -1)
            concat_f = torch.cat([cls_t, all_token], dim=-1)
            ret = self.cls_head_finetune(concat_f)
            return ret
        elif self.cls_head == 50:  # all cls  + all token  384
            x, cls_x = self.blocks(x, pos)
            cls_x = torch.cat(cls_x, dim=-1)
            all_token = x[:, 1:].reshape(batch_size, -1)
            concat_f = torch.cat([cls_x, all_token], dim=-1)
            ret = self.cls_head_finetune(concat_f)
            return ret



class MaskTransformer(nn.Module):
    def __init__(self, model_type, config, **kwargs):
        self.model_type = model_type
        super().__init__()
        self.config = config

        # define the encoder
        self.num_group = config.transformer_config.num_group
        self.group_size = config.transformer_config.group_size
        self.encoder_dims = config.transformer_config.encoder_dims

        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio
        # self.mask_ratio_rand = config.transformer_config.mask_ratio_rand
        # self.mask_ratio_block =  config.transformer_config.mask_ratio_block

        self.trans_dim = config.transformer_config.trans_dim

        self.use_bn_fc = config.transformer_config.use_bn_fc
        self.hidden_dim = config.transformer_config.hidden_dim
        self.output_dim_fc = config.transformer_config.output_dim_fc

        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.cls_dim = config.transformer_config.cls_dim
        self.replace_pob = config.transformer_config.replace_pob
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[Transformer args] {config.transformer_config}', logger='dVAE BERT')
        # define the encoder
        self.encoder_dims = config.transformer_config.encoder_dims
        # self.encoder = Encoder(encoder_channel = self.encoder_dims)

        self.enc_arch = config.transformer_config.get('enc_arch', 'Point_ViT')
        if self.enc_arch == '3detr':
            self.encoder = build_preencoder_3detr(num_group=self.num_group, group_size=self.group_size,
                                                  dim=self.encoder_dims)
        else:
            self.encoder = Encoder(encoder_channel=self.encoder_dims)

        # bridge encoder and transformer
        self.reduce_dim = nn.Identity()
        if self.encoder_dims != self.trans_dim:
            self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        try:
            self.mask_type = config.transformer_config.mask_type
        except:
            self.mask_type = False

        # define the learnable tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        # pos embedding for each patch
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        # -- projection head
        self.p_head = None
        p_head = OrderedDict([])
        p_head['fc1'] = torch.nn.Linear(self.trans_dim, self.hidden_dim)

        if self.use_bn_fc:
            p_head['bn1'] = torch.nn.BatchNorm1d(self.hidden_dim)
        p_head['gelu1'] = torch.nn.GELU()
        p_head['fc2'] = torch.nn.Linear(self.hidden_dim, self.hidden_dim)

        if self.use_bn_fc:
            p_head['bn2'] = torch.nn.BatchNorm1d(self.hidden_dim)
        p_head['gelu2'] = torch.nn.GELU()
        p_head['fc3'] = torch.nn.Linear(self.hidden_dim, self.output_dim_fc)
        self.p_head = torch.nn.Sequential(p_head)

        # define the transformer blocks
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        if self.enc_arch == '3detr':
            self.blocks = build_encoder_3detr(
                ndim=self.trans_dim,
                nhead=self.num_heads,
                nlayers=self.depth
            )
        else:
            self.blocks = TransformerEncoder(
                embed_dim=self.trans_dim,
                depth=self.depth,
                drop_path_rate=dpr,
                num_heads=self.num_heads
            )
        # self.blocks = TransformerEncoder(
        #     embed_dim = self.trans_dim,
        #     depth = self.depth,
        #     drop_path_rate = dpr,
        #     num_heads = self.num_heads
        # )
        # layer norm
        self.norm = nn.LayerNorm(self.trans_dim)

        # initialize the learnable tokens
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        trunc_normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def makeRandomRate(self):
        # test  62 用
        rand_num = np.random.random()
        start_rate = 0.4
        end_rate = 0.8

        rate = (end_rate - start_rate) * rand_num + start_rate

        return rate

    def preencoder(self, neighborhood):
        group_input_tokens = self.encoder(neighborhood)  # B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        return group_input_tokens


    def _mask_center_block(self, center, noaug=False, mask_Rate=0):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or mask_Rate == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []



        ratio = mask_Rate

        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G


            mask_num = int(ratio * len(idx))

            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand_mask_l(self, center, mask_ratio, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        # mask_ratio = self.config.transformer_config.mask_ratio
        #
        # # self.num_mask = torch.tensor(mask_ratio * G, dtype=torch.int64, requires_grad=True)
        # self.num_mask = torch.tensor(mask_ratio * G, requires_grad=True).round()
        #
        # torch.ones(self.num_mask)

        mask_ratio = mask_ratio.round()
        mask_ratio_my = torch.sum(mask_ratio) / 64

        overall_mask = torch.zeros(B, G)
        for i in range(B):
            mask = torch.hstack([
                mask_ratio
            ])
            # torch.random.shuffle(mask)

            b = torch.randperm(mask.size(0))
            mask = mask[b]

            overall_mask[i, :] = mask
        # overall_mask = overall_mask.to(torch.bool)

        return overall_mask.to(center.device)  # B G

    def _mask_center_rand(self, center, noaug=False, mask_Rate=0):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or mask_Rate == 0:
            return torch.zeros(center.shape[:2]).bool()

        mask_ratio = mask_Rate

        self.num_mask = int(mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G

    def _random_replace(self, group_input_tokens, bool_masked_pos, noaug=False):
        '''
            group_input_tokens : B G C
            bool_masked_pos : B G
            -----------------
            replaced_group_input_tokens: B G C
        '''
        # skip replace
        if noaug or self.replace_pob == 0:
            return group_input_tokens, bool_masked_pos

        replace_mask = (torch.rand(group_input_tokens.shape[:2]) < self.replace_pob).to(bool_masked_pos.device).bool()
        replace_mask = (replace_mask & ~bool_masked_pos)  # do not replace the mask pos

        overall_mask = (replace_mask + bool_masked_pos).bool().to(bool_masked_pos.device)  # True for flake input

        detached_group_input_tokens = group_input_tokens.detach()
        flatten_group_input_tokens = detached_group_input_tokens.reshape(
            detached_group_input_tokens.size(0) * detached_group_input_tokens.size(1),
            detached_group_input_tokens.size(2))
        idx = torch.randperm(flatten_group_input_tokens.shape[0])
        shuffled_group_input_tokens = flatten_group_input_tokens[idx].reshape(detached_group_input_tokens.size(0),
                                                                              detached_group_input_tokens.size(1),
                                                                              detached_group_input_tokens.size(2))

        replace_mask = replace_mask.unsqueeze(-1).type_as(detached_group_input_tokens)
        replaced_group_input_tokens = group_input_tokens * (
                    1 - replace_mask) + shuffled_group_input_tokens * replace_mask
        return replaced_group_input_tokens, overall_mask

    def forward(self, neighborhood, center, mask_type, mask_Rate, noaug=False):

        if self.model_type is 'pc_msn_target_encoder':

            # group_input_tokens = self.encoder(neighborhood)  # B G N

            if self.enc_arch == '3detr':
                pre_enc_xyz, group_input_tokens, pre_enc_inds = self.preencoder(center)
                group_input_tokens = group_input_tokens.permute(0, 2, 1)
                center = pre_enc_xyz
            else:
                group_input_tokens = self.encoder(neighborhood)  # B G N

            batch_size, seq_len, C = group_input_tokens.size()

            # add pos embedding
            pos = self.pos_embed(center)

            if self.enc_arch == '3detr':
                # 没输入cls  token
                x = self.blocks(group_input_tokens.transpose(0, 1), pos=pos.transpose(0, 1))[1].transpose(0, 1)

                x_vis = x
                cls_feat = self.p_head(torch.mean(x_vis, dim=1))
                return cls_feat, 1, 1

            else:
                # prepare cls and mask
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                cls_pos = self.cls_pos.expand(batch_size, -1, -1)

                # final input
                x = torch.cat((cls_tokens, group_input_tokens), dim=1)
                pos = torch.cat((cls_pos, pos), dim=1)

                # transformer
                x, _ = self.blocks(x, pos)
                x = self.norm(x)
                cls_feat = x[:, 0]
                cls_feat = self.p_head(cls_feat)
                return cls_feat, _, _


        elif self.model_type is 'pc_msn_encoder':

            if self.enc_arch == '3detr':
                pre_enc_xyz, group_input_tokens, pre_enc_inds = self.preencoder(center)
                group_input_tokens = group_input_tokens.permute(0, 2, 1)
                center = pre_enc_xyz
            else:
                group_input_tokens = self.encoder(neighborhood)  # B G N

            if mask_type == 'rand':
                bool_masked_pos = self._mask_center_rand(center, noaug=noaug, mask_Rate=mask_Rate)
            if mask_type == 'block':
                bool_masked_pos = self._mask_center_block(center, noaug=noaug, mask_Rate=mask_Rate)

            batch_size, seq_len, C = group_input_tokens.size()

            # learned mask ratio
            # x_vis = group_input_tokens[bool_masked_pos].reshape(batch_size, -1 , C)
            x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
            # x_vis = group_input_tokens.reshape(batch_size, -1, C)

            # add pos embedding
            # mask pos center
            masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
            # masked_center = center.reshape(batch_size, -1, 3)
            pos = self.pos_embed(masked_center)

            masked_input_tokens = x_vis
            if self.enc_arch == '3detr':
                # 没输入cls  token
                x = self.blocks(masked_input_tokens.transpose(0, 1), pos=pos.transpose(0, 1))[1].transpose(0, 1)

                x_vis = x
                cls_feat = self.p_head(torch.mean(x_vis, dim=1))
                return cls_feat, x_vis, bool_masked_pos, center

            else:
                # prepare cls and mask
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)

                cls_pos = self.cls_pos.expand(batch_size, -1, -1)

                # final input
                x_vis = torch.cat((cls_tokens, x_vis), dim=1)
                pos = torch.cat((cls_pos, pos), dim=1)

                # transformer
                x_vis, all_cls = self.blocks(x_vis, pos)
                x_vis = self.norm(x_vis)
                cls_feat = x_vis[:, 0]
                cls_feat = self.p_head(cls_feat)

                return cls_feat, x_vis[:, 1:], bool_masked_pos

            # if noaug:
            #     # return x_vis, all_cls
            #     return x_vis, _


@MODELS.register_module()
class Point_BERT(nn.Module):
    def __init__(self, config):

        super().__init__()

        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')

        self.config = config
        self.m = config.m
        self.loss_func = ChamferDistanceL2().cuda()
        self.num_group = config.transformer_config.num_group
        self.trans_dim = config.transformer_config.trans_dim
        self.group_size = config.transformer_config.group_size
        self.total_bs = config.transformer_config.total_bs
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_type = config.transformer_config.mask_type
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )
        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.mask_ratio_learned = nn.Parameter(torch.rand(3, self.num_group))

        self.pc_msn_encoder = MaskTransformer('pc_msn_encoder', config)

        self.pc_msn_target_encoder = MaskTransformer('pc_msn_target_encoder', config)
        for param_q, param_k in zip(self.pc_msn_encoder.parameters(), self.pc_msn_target_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.group_size = config.transformer_config.group_size
        self.num_group = config.transformer_config.num_group

        print_log(
            f'[Point_LGMask Group]  divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger='Point_LGMask')

        self.enc_arch = config.transformer_config.enc_arch
        self.group_divider = (DummyGroup if self.enc_arch == '3detr' else Group)(num_group=self.num_group,
                                                                                 group_size=self.group_size)

        if self.enc_arch == '3detr':
            self.get_neigh = Group_maskpoint_only_neig(group_size=self.group_size)
        # self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        self.group_divider_crop = Crop_pc(num_group=self.num_group, group_size=self.group_size)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.pc_msn_encoder.parameters(), self.pc_msn_target_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def get_maskedPC(self, neighborhood, center, mask):

        bool_masked_pos = mask

        batch_size, patch_num, pc_num, C = neighborhood.size()

        # vis_patch = neighborhood[~bool_masked_pos].reshape(batch_size, -1, pc_num, C)
        # vis_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        # maskedPC = vis_patch + vis_center.unsqueeze(2)

        vis_patch_1 = neighborhood[~bool_masked_pos].reshape(batch_size, -1, pc_num, C)
        vis_center_1 = center[~bool_masked_pos].reshape(batch_size, -1, 1, 3)
        vis_center_1 = vis_center_1.repeat(1, 1, pc_num, 1)
        maskedPC_1 = vis_patch_1 + vis_center_1

        return maskedPC_1.reshape(batch_size, -1, 3)

    def get_maskedPC_2(self, original_neigh, mask):

        bool_masked_pos = mask

        batch_size, patch_num, pc_num, C = original_neigh.size()

        vis_patch_1 = original_neigh[~bool_masked_pos].reshape(batch_size, -1, pc_num, C)

        return vis_patch_1.reshape(batch_size, -1, 3)

    def get_predPC(self, neighborhood, center, x_vis, mask):

        B, _, C = x_vis.shape  # B VIS C

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)

        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N)

        B, M, C = x_rec.shape
        pred_Pc = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B, M, -1, 3)
        masked_center = center[mask].reshape(B, M, 3)
        pred_Pc = pred_Pc + masked_center.unsqueeze(2)

        return pred_Pc.reshape(B, -1, 3)

    def get_one_mask_pre_pc_pair(self, neighborhood, center, original_neigh, mask_type='rand', mask_Rate=0.3):
        _, x_vis, mask = self.pc_msn_encoder(neighborhood, center, mask_type=mask_type, mask_Rate=mask_Rate)

        # masked_pc = self.get_maskedPC(neighborhood, center, mask)
        masked_pc = self.get_maskedPC_2(original_neigh, mask)

        pred_pc = self.get_predPC(neighborhood, center, x_vis, mask)

        full_Pc = torch.cat((masked_pc, pred_pc), 1)

        return masked_pc, full_Pc

    def forward_vis(self, pts):
        masked_pc = []
        re_Pc = []

        with torch.no_grad():
            neighborhood, center, original_neigh = self.group_divider(pts, visFalg=1)
            masked_pc_1, full_Pc_1 = self.get_one_mask_pre_pc_pair(neighborhood, center, original_neigh, mask_Rate=0.15)
            masked_pc.append(masked_pc_1)
            re_Pc.append(full_Pc_1)

            masked_pc_2, full_Pc_2 = self.get_one_mask_pre_pc_pair(neighborhood, center, original_neigh, mask_Rate=0.3)
            masked_pc.append(masked_pc_2)
            re_Pc.append(full_Pc_2)

            masked_pc_3, full_Pc_3 = self.get_one_mask_pre_pc_pair(neighborhood, center, original_neigh, mask_Rate=0.45)
            masked_pc.append(masked_pc_3)
            re_Pc.append(full_Pc_3)

            masked_pc_4, full_Pc_4 = self.get_one_mask_pre_pc_pair(neighborhood, center, original_neigh, mask_Rate=0.6)
            masked_pc.append(masked_pc_4)
            re_Pc.append(full_Pc_4)

            masked_pc_5, full_Pc_5 = self.get_one_mask_pre_pc_pair(neighborhood, center, original_neigh, mask_Rate=0.75)
            masked_pc.append(masked_pc_5)
            re_Pc.append(full_Pc_5)

            masked_pc_6, full_Pc_6 = self.get_one_mask_pre_pc_pair(neighborhood, center, original_neigh, mask_Rate=0.9)
            masked_pc.append(masked_pc_6)
            re_Pc.append(full_Pc_6)

            return pts, masked_pc, re_Pc

    def forward_eval(self, pts):
        with torch.no_grad():
            neighborhood, center = self.group_divider(pts)
            all_tokens, all_cls = self.pc_msn_encoder(neighborhood, center, mask_type='rand', mask_Rate=0.3, noaug=True)
            # # B = all_tokens.shape[0]
            # # all_tokens = all_tokens.reshape(B,-1)
            # cls_x = torch.cat((all_cls[2], all_cls[5], all_cls[8], all_cls[11]), dim=-1)
            #
            # x  = all_tokens
            # all_tokens = torch.cat((cls_x,x[:, 1:].max(1)[0], x[:, 1:].mean(1)), dim=-1)

            # print("all_tokens",all_tokens.shape)
            # return all_tokens

            # return torch.cat((all_tokens[:, 0], all_tokens[:, 1:].max(1)[0]), dim=-1)
            return torch.cat((all_tokens[:, 0], all_tokens[:, 1:].max(1)[0], all_tokens[:, 1:].mean(1)), dim=-1)

    def rec_loss(self, neighborhood, center, x_vis, mask):
        B, _, C = x_vis.shape  # B VIS C

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)

        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        # test time
        # x_vis = x_vis[~mask].reshape(B, -1, C)

        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N)

        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

        gt_points = neighborhood[mask].reshape(B * M, -1, 3)
        loss1 = self.loss_func(rebuild_points, gt_points)

        return loss1



    # for Point-LGMask
    def forward(self, pts, noaug=False, vis_flag=0, **kwargs):
        if vis_flag == 1:
            return self.forward_vis(pts)
        elif noaug:
            return self.forward_eval(pts)
        else:

            pts_target = train_transforms_1(pts.clone())
            pts_anchor = train_transforms_2(pts)

            # anchor_views
            neighborhood, center = self.group_divider(pts_anchor)

            anchor_views_1, x_vis, mask  = self.pc_msn_encoder(neighborhood, center, mask_type=self.mask_type, mask_Rate=0.3)
            re_loss_1 = self.rec_loss(neighborhood, center, x_vis, mask)
            anchor_views_1  = nn.functional.normalize(anchor_views_1 , dim=1)

            anchor_views_2, x_vis, mask  = self.pc_msn_encoder(neighborhood, center, mask_type=self.mask_type, mask_Rate=0.6)
            re_loss_2 = self.rec_loss(neighborhood, center, x_vis, mask)
            anchor_views_2  = nn.functional.normalize(anchor_views_2 , dim=1)
            anchor_views = torch.cat((anchor_views_1, anchor_views_2), 0)

            anchor_views_3, x_vis, mask  = self.pc_msn_encoder(neighborhood, center, mask_type=self.mask_type, mask_Rate=0.9)
            re_loss_3 = self.rec_loss(neighborhood, center, x_vis, mask)
            anchor_views_3  = nn.functional.normalize(anchor_views_3 , dim=1)
            anchor_views = torch.cat((anchor_views, anchor_views_3), 0)

            all_re_loss = (re_loss_1 + re_loss_2 + re_loss_3)/3


            # compute  and  update target_encoder and
            with torch.no_grad():  # no gradient to keys
                # target_views
                neighborhood_2, center_2 = self.group_divider(pts_target)

                # mask_t='1' denote that no use mask
                target_views, _, _ = self.pc_msn_target_encoder(neighborhood_2, center_2, mask_type='1', mask_Rate=0.3)
                target_views = nn.functional.normalize(target_views, dim=1)
                self._momentum_update_key_encoder()  # update the key encoder

            anchor_views, target_views = anchor_views, target_views.detach()

            return anchor_views, target_views, all_re_loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output