import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple
from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger
from mmcv.runner.checkpoint import load_state_dict


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, windowsize=7):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.windowsize = windowsize
        self.sr_ratio = sr_ratio

        if sr_ratio > 1:
            self.mul = [2, 7, 7]
        if sr_ratio == 1:
            self.mul = [1, 7]

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.single_dim = []
        self.scales = []

        for i_layer in range(len(self.mul)):

            single_d = dim // sum(self.mul) * self.mul[i_layer]
            self.single_dim.append(single_d)

            if i_layer == 0:
                scale = single_d ** -0.5
            else:
                scale = (single_d //  self.num_heads) ** -0.5
            self.scales.append(scale)
        
        self.kv_g = nn.Linear(self.single_dim[0], self.single_dim[0] * 2, bias=qkv_bias)
        self.kv_l = DEPTHWISECONV(dim - self.single_dim[0], dim - self.single_dim[0])

        self.sr = nn.Conv2d(self.single_dim[0], self.single_dim[0], kernel_size=sr_ratio, stride=sr_ratio)
        self.norm = nn.LayerNorm(self.single_dim[0])
        self.local_conv_g = nn.Conv2d(self.single_dim[0], self.single_dim[0], kernel_size=3, padding=1, stride=1, groups=self.single_dim[0])  
        
        self.unfolds = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.single_heads = nn.ModuleList()
        self.local_convs = nn.ModuleList()
        self.strides = []

        for i_layer in range(1, len(self.mul)):
            if i_layer == 1:
                dilation = 1
            else:
                dilation = self.sr_ratio
            kernel_size = self.windowsize
            stride = dilation * (kernel_size - 1) + 1
            self.strides.append(stride)

            unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation)
            fc = nn.Linear(self.single_dim[i_layer] // self.num_heads, self.single_dim[i_layer] // self.num_heads, bias=qkv_bias)
            single = nn.Linear(self.single_dim[i_layer] // self.num_heads, 2 * self.single_dim[i_layer] // self.num_heads, bias=qkv_bias)
            local_conv = nn.Conv2d(self.single_dim[i_layer], self.single_dim[i_layer], kernel_size=3, padding=1, stride=1, groups=self.single_dim[i_layer])
            self.unfolds.append(unfold)
            self.fcs.append(fc)
            self.single_heads.append(single)
            self.local_convs.append(local_conv)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x)   #B,N,C - B,N,C

        q_g = q[:,:,:self.single_dim[0]].reshape(B, N, 1, self.single_dim[0] // 1).permute(0, 2, 1, 3)   #B,N,C_g - B,N,h,C_g/h - B,h,N,C_g/h
        x_g = x[:,:,:self.single_dim[0]].permute(0, 2, 1).reshape(B, self.single_dim[0], H, W)   #B,N,C_g - B,C_g,N - B,C_g,H,W
        x_g = self.sr(x_g).reshape(B, self.single_dim[0], -1).permute(0, 2, 1)    #B,C_g,H,W - B,C_g,H_g,W_g - B,C_g,N_g - B,N_g,C_g
        x_g = self.norm(x_g)    #B,N_g,C_g - B,N_g,C_g
        kv_g = self.kv_g(x_g).reshape(B, -1, 2, 1, self.single_dim[0] // 1).permute(2, 0, 3, 1, 4)    #B,N_g,C_g - B,N_g,C_g*2 - B,N_g,2,h,C_g/h - 2,B,h,N_g,C_g/h
        k_g, v_g = kv_g[0], kv_g[1] #2,B,h,N_g,C_g/h - B,h,N_g,C_g/h
        attn_g = (q_g @ k_g.transpose(-2, -1)) * self.scales[0] #B,h,N,C_g/h;B,h,N_g,C_g/h - B,h,N,N_g
        attn_g = attn_g.softmax(dim=-1)
        attn_g = self.attn_drop(attn_g)
        v_g = v_g + self.local_conv_g(v_g.transpose(1, 2).reshape(B, -1, self.single_dim[0]).transpose(1, 2).view(B,self.single_dim[0], H//self.sr_ratio, W//self.sr_ratio)).view(B, self.single_dim[0], -1).view(B, 1, self.single_dim[0] // 1, -1).transpose(-1, -2)
        #B,h,N_g,C_g/h - B,N_g,h,C_g/h - B,N_g,C_g - B,C_g,N_g - B,C_g,H_g,W_g - B,C_g,H_g,W_g - B,C_g,N_g - B,h,C_g/h,N_g - B,h,N_g,C_g/h -B,h,N_g,C_g/h
        attn_g = (attn_g @ v_g).transpose(1, 2).reshape(B, N, self.single_dim[0])    #B,h,N,N_g;B,h,N_g,C_g/h - B,h,N,C_g/h - B,N,h,C_g/h - B,N,C_g


        q_l = q[:,:,self.single_dim[0]:].reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)    #B,N,C_l - B,N,h,C_l/h - B,h,N,C_l/h
        x_l = x[:,:,self.single_dim[0]:].reshape(B, H, W, -1).permute(0, 3, 1, 2) #B,N,C_l - B,H,W,C_l - B,C_l,H,W
        kv_l = self.kv_l(x_l) #B,C_l,H,W - B,C_l,H,W
        attn_l = []
        for i_c in range(1, len(self.mul)):
            if i_c == 1:
                q_ = q_l[:,:,:,:self.single_dim[1]//self.num_heads]   #B,h,N,C_l_i/h
                kv_ = kv_l[:,:self.single_dim[1],:,:]   #B,C_l_i,H,W
            else:
                q_ = q_l[:,:,:,self.single_dim[1]//self.num_heads:]   #B,h,N,C_l_i/h
                kv_ = kv_l[:,self.single_dim[1]:,:,:]   #B,C_l_i,H,W
            pad_l = pad_t = 0
            pad_r = (self.strides[i_c-1] - W % self.strides[i_c-1]) % self.strides[i_c-1]
            pad_b = (self.strides[i_c-1] - H % self.strides[i_c-1]) % self.strides[i_c-1]
            rp = nn.ReflectionPad2d((pad_l, pad_r, pad_t, pad_b))
            kv_ = rp(kv_)   #B,C_l_i,H,W - #B,C_l_i,Hi,Wi

            kv_ = self.unfolds[i_c-1](kv_) #B,C_l_i,Hi,Wi - B,C_l_i*L2,H_i*W_i
            kv_ = kv_.reshape(B, self.num_heads, self.single_dim[i_c] // self.num_heads, self.windowsize**2, -1).permute(0,1,3,4,2)
            #B,C_l_i*L2,H_i*W_i - B,h,C_l_i/h,L2,H_i*W_i - B,h,L2,H_i*W_i,C_l_i/h
            kv_ = kv_.reshape(B,self.num_heads,self.windowsize**2,-1) #B,h,L2,H_i*W_i,C_l_i/h - B,h,L2,H_i*W_i*C_l_i/h
            kv_ = nn.AdaptiveAvgPool2d((None, self.single_dim[i_c] // self.num_heads))(kv_) #B,h,L2,H_i*W_i*C_l_i/h - B,h,L2,C_l_i/h
            kv_ = self.fcs[i_c-1](kv_)  #B,h,L2,C_l_i/h - B,h,L2,C_l_i/h
            kv_ = self.single_heads[i_c-1](kv_)    #B,h,L2,C_l_i/h - B,h,L2,2C_l_i/h
            k_ = kv_[:,:,:,:(self.single_dim[i_c] // self.num_heads)] #B,h,L2,C_l_i/h
            v_ = kv_[:,:,:,(self.single_dim[i_c] // self.num_heads):]

            attn_ = (q_ @ k_.transpose(-2, -1)) * self.scales[i_c]  #B,h,N,C_l_i/h;B,h,L2,C_l_i/h - B,h,N,L2
            attn_ = attn_.softmax(dim=-1)
            attn_ = self.attn_drop(attn_)
            v_ = v_ + self.local_convs[i_c-1](v_.transpose(2, 3).reshape(B,self.single_dim[i_c],self.windowsize,self.windowsize)).view(B, self.num_heads, self.single_dim[i_c] // self.num_heads, self.windowsize**2).transpose(2, 3)
            #B,h,L2,C_l_i/h - B,h,C_l_i/h,L2 - B,C_l_i,L,L - B,C_l_i,L,L - B,h,C_l_i/h,L2 - B,h,L2,C_l_i/h
            attn_ = (attn_ @ v_).permute(0, 2, 1, 3) #B,h,N,L2;B,h,L2,C_l_i/h - B,h,N,C_l_i/h - B,N,h,C_l_i/h
            attn_ = attn_.reshape(B, N, self.single_dim[i_c])  #B,N,h,C_l_i/h - B,N,C_l_i
            attn_l.append(attn_)
        
        for i_c in range(1,len(self.mul)):
            attn_g = torch.cat((attn_g,attn_l[i_c-1]),-1)

        x = self.proj(attn_g)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        patch_size = to_2tuple(patch_size)
        
        assert max(patch_size) > stride, "Set larger patch_size than stride"
        
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class FactorizationTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[8, 6, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=4, pretrained=None):
        super(FactorizationTransformer, self).__init__()
        # self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.pretrained = pretrained
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def init_weights(self, map_location='cpu'):
        if isinstance(self.pretrained, str):
            logger = get_root_logger()

            checkpoint = torch.load(self.pretrained, map_location=map_location)
            checkpoint = checkpoint['model']
            # OrderedDict is a subclass of dict
            if not isinstance(checkpoint, dict):
                raise RuntimeError(
                    f'No state_dict found in checkpoint file {self.pretrained}')
            # get state_dict from checkpoint
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # load state_dict
            load_state_dict(self, state_dict, strict=False, logger=logger)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs


    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class DEPTHWISECONV(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DEPTHWISECONV, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@BACKBONES.register_module()
class favit_b0(FactorizationTransformer):
    def __init__(self, **kwargs):
        super(favit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 128, 256], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 6, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 6, 2],
            drop_rate=0.0, drop_path_rate=0.1, pretrained=kwargs['pretrained'])


@BACKBONES.register_module()
class favit_b1(FactorizationTransformer):
    def __init__(self, **kwargs):
        super(favit_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 6, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 6, 2],
            drop_rate=0.0, drop_path_rate=0.1, pretrained=kwargs['pretrained'])


@BACKBONES.register_module()
class favit_b2(FactorizationTransformer):
    def __init__(self, **kwargs):
        super(favit_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 6, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 3, 18, 3],
            drop_rate=0.0, drop_path_rate=0.1, pretrained=kwargs['pretrained'])


@BACKBONES.register_module()
class favit_b3(FactorizationTransformer):
    def __init__(self, **kwargs):
        super(favit_b3, self).__init__(
            patch_size=4, embed_dims=[96, 192, 384, 768], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 6, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 3, 14, 3],
            drop_rate=0.0, drop_path_rate=0.1, pretrained=kwargs['pretrained'])