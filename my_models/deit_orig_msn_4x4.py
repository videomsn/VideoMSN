# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from functools import partial

# from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg
from timm.models.vision_transformer import Mlp, _cfg

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from einops import rearrange

import math
import numpy as np

import torch.nn.functional as F

class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x 
    
class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class Layer_scale_init_Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_1_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        
    def forward(self, x):
        x = x + self.drop_path(self.gamma_1*self.attn(self.norm1(x))) + self.drop_path(self.gamma_1_1 * self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x))) + self.drop_path(self.gamma_2_1 * self.mlp1(self.norm21(x)))
        return x
        
class Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.mlp1(self.norm21(x)))
        return x
        
        
class hMLP_stem(nn.Module):
    """ hMLP_stem: https://arxiv.org/pdf/2203.09795.pdf
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, embed_dim=768,norm_layer=nn.SyncBatchNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = torch.nn.Sequential(*[nn.Conv2d(in_chans, embed_dim//4, kernel_size=4, stride=4),
                                          norm_layer(embed_dim//4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim//4, embed_dim//4, kernel_size=2, stride=2),
                                          norm_layer(embed_dim//4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim//4, embed_dim, kernel_size=2, stride=2),
                                          norm_layer(embed_dim),
                                         ])
        

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
class vit_models(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, num_classes=400, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers = Block,
                 Patch_layer=PatchEmbed,act_layer=nn.GELU,
                 Attention_block = Attention, Mlp_block=Mlp,
                dpr_constant=True,init_scale=1e-4,
                mlp_ratio_clstk = 4.0,
                duration=8,    ##added for super images 
                super_img_rows=1,
                patch_drop=0.5,
                **kwargs):
        super().__init__()
        
        self.dropout_rate = drop_rate

            
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale)
            for i in range(depth)])
        

        
            
        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        
        self.img_size=img_size
        self.duration=duration
        self.super_img_rows=super_img_rows
        self.patch_drop=patch_drop

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def pad_frames(self, x):
        frame_num = self.duration - self.frame_padding
        x = x.view((-1,3*frame_num)+x.size()[2:])
        x_padding = torch.zeros((x.shape[0], 3*self.frame_padding) + x.size()[2:]).cuda()
        x = torch.cat((x, x_padding), dim=1)
        assert x.shape[1] == 3 * self.duration, 'frame number %d not the same as adjusted input size %d' % (x.shape[1], 3 * self.duration)

        return x
    
    def create_super_img(self, x):
        input_size = x.shape[-2:]
        if input_size != to_2tuple(self.img_size):
            x = nn.functional.interpolate(x, size=self.img_size,mode='bilinear')
        x = rearrange(x, 'b (th tw c) h w -> b c (th h) (tw w)', th=self.super_img_rows, c=3)
        return x

    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)
    
    def save_image(self,x,add):
        from PIL import Image
        k=0
        for ik, i in enumerate(x):
            images = i.cpu().numpy()
            super=images.transpose(1,2,0)
            super=Image.fromarray((super*255).astype(np.uint8))
            super.save(add+str(ik)+'.png')

    # def interpolate_pos_encoding(self, x, pos_embed):
    #     npatch = x.shape[1] - 1
    #     N = pos_embed.shape[1] - 1
    #     if npatch == N:
    #         return pos_embed
    #     class_emb = pos_embed[:, 0]
    #     pos_embed = pos_embed[:, 1:]
    #     dim = x.shape[-1]
    #     pos_embed = nn.functional.interpolate(
    #         pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
    #         scale_factor=math.sqrt(npatch / N),
    #         mode='bicubic',
    #     )
    #     pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    #     return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)
    
    def interpolate_pos_embed(self, x, pos_embed):
        # pos_embed = model.pos_embed  # Shape: [1, 196, hidden_dim]
    
        # Get the original and new grid sizes
        num_patches_old = int(self.pos_embed.shape[1] ** 0.5)   #1764 42 #int(pos_embed.shape[1] ** 0.5)  # 14 for 224x224
        num_patches_new = int(x.shape[1] ** 0.5)  #324  #int(new_size / model.patch_embed.patch_size)  # 42 for 672x672
    
        # print("num_patches_old:",num_patches_old)
        # print("num_patches_new:",num_patches_new)
        # if num_patches_new!=1764:
        #     exit(0)
    
        if num_patches_old != num_patches_new:
            # print(f"Interpolating position embeddings from {num_patches_old}x{num_patches_old} to {num_patches_new}x{num_patches_new}")
    
            pos_embed = pos_embed.reshape(1, num_patches_old, num_patches_old, -1)  # Reshape to [1, 14, 14, hidden_dim]
            pos_embed = pos_embed.permute(0, 3, 1, 2)  # [1, hidden_dim, 14, 14]
    
            # Interpolate
            pos_embed = F.interpolate(pos_embed, size=(num_patches_new, num_patches_new), mode='bicubic', align_corners=False)
    
            # Reshape back
            pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, num_patches_new * num_patches_new, -1)  # [1, 1764, hidden_dim]
    
            # print("pos_embed.shape:",pos_embed.shape[0]*pos_embed.shape[1]*pos_embed.shape[2])
            # print("pos_embed..uninque.shape:",torch.unique(pos_embed).shape)
            # tensor_reshaped = pos_embed.squeeze(0)
            # unique_rows = torch.unique(tensor_reshaped, dim=0)
            # is_unique = unique_rows.shape[0] == tensor_reshaped.shape[0]
            # print(f"Are all 384-length tensors unique? {is_unique}")
            # exit(0)
    
            # model.pos_embed = torch.nn.Parameter(pos_embed)
    
        return pos_embed

    def forward_features(self, x, msn_pretraining=False):
        
        B = x.shape[0]
        
        # print("B:",B)
        # print("x.shape:",x.shape)
        
        
        # self.frame_padding = self.duration % super_img_rows
        # if self.frame_padding != 0:
        #     self.frame_padding = self.super_img_rows - self.frame_padding
        #     self.duration += self.frame_padding

        # self.frame_padding = 1
        # self.duration = 9
        # self.image_mode = True
        # self.img_size = 224
        # self.super_img_rows = 3
        # print("self.frame_padding:",self.frame_padding)
        # print("self.duration:",self.duration)
        # print("self.image_mode:",self.image_mode)
        # print("x.shape:",x.shape) 
        
        # if self.frame_padding > 0:
        #     x = self.pad_frames(x)
        # else:
        #     x = x.view((-1,3*self.duration)+x.size()[2:])


        # x = self.create_super_img(x)

            
            
        x = self.patch_embed(x)
        
        # print("self.cls_token.shape:",self.cls_token.shape)
        # print("self.pos_embed.shape:",self.pos_embed.shape)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # print("x.shape",x.shape)
        # print("self.pos_embed",self.pos_embed.shape)
        pos_embed=self.interpolate_pos_embed(x, self.pos_embed)
        # print("pos_embed:",pos_embed.shape)
        # print("After self.pos_embed",self.pos_embed.shape)
        # print("x.shape",x.shape)
        # print("self.pos_embed",self.pos_embed.shape)
        x = x + pos_embed
        # print("x.shape",x.shape)
        # print("cls_tokens.shape:",cls_tokens.shape)
        # exit(0)
        x = torch.cat((cls_tokens, x), dim=1)
        # pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        
        
        # ===== ADDED  (START) FOR PATCH DROPPING =====
        if msn_pretraining:
            patch_drop = self.patch_drop 
            num_frames = self.duration
            
            # print("patch_drop:",patch_drop)
            # print("num_frames:",num_frames)
            
            # exit(0)
            
            patch_keep = 1. - patch_drop
            
            new_num_patches= x.shape[1] -1
            num_patches_per_frame = (x.shape[1] -1)//num_frames
            all_patch_idx = torch.reshape(torch.arange(1, x.shape[1], dtype = torch.int),(num_frames,num_patches_per_frame))
            # Number of patches to keep in each frame
            num_kept_patches = int(np.floor(num_patches_per_frame*patch_keep))
            # ids of the patches in each frame to keep
            kept_patches_idx = torch.randperm(num_patches_per_frame)[:num_kept_patches]
            sorted_kept_patches_idx, _ = torch.sort(kept_patches_idx)
            all_kept_patches_idx = all_patch_idx[:,sorted_kept_patches_idx].flatten()
            all_kept_patches_idx = torch.cat([torch.zeros(1, dtype=all_kept_patches_idx.dtype, device=all_kept_patches_idx.device), all_kept_patches_idx])
            # Extract only the patches from x corresponding to the kept idx
            x = x[:,all_kept_patches_idx,:]
        # ===== ADDED (END) FOR PATCH DROPPING =====
        
        for i , blk in enumerate(self.blocks):
            x = blk(x)
            
        x = self.norm(x)
        
        x = x[:, 0]
        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)
        
        return x
    
        # cls_tokens = self.cls_token.expand(B, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        # pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        # x = x + pos_embed
        # x = self.pos_drop(x)

    def forward(self, x, msn_pretraining=False, temporal_aug=False):
        
        
        # print(len(x))                   #7
        if not isinstance(x, list):
            target_x = rearrange(x, 'b (th tw c) h w -> b c (th h) (tw w)', th=4, c=3)
            h = self.forward_features(target_x, False)
            # print("Inside forward Target H:",h.shape)
        else:
            #print("x[0].shape",x[0].shape)  #torch.Size([2, 27, 224, 224])
            #print("x[1].shape",x[1].shape)  #torch.Size([2, 27, 96, 96])
            #print("x[6].shape",x[6].shape)  #torch.Size([2, 27, 96, 96])
            
            idx_crops = torch.cumsum(torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1], 0)            #1,7
            # print("idx_crops:",idx_crops[0].item()) #.item() → Converts the single-element tensor into a Python integer.
            
            rand_views=idx_crops[0].item()
            
            # print("x[0].size:",x[0].size())
            # print("x[1:].size:",torch.cat(x[1:]).size())
            
            rand_x = rearrange(x[0], 'b (th tw c) h w -> b c (th h) (tw w)', th=4, c=3) #[2, 27, 224, 224]-->[2 3 672 672]    for 4x4: [1, 48, 224, 224] --> [1, 3, 896, 896]
            if temporal_aug:
                focal_x = rearrange(torch.cat(x[rand_views:]), 'b (th tw c) h w -> b c (th h) (tw w)', th=3, c=3) #[12 27 96 96] --> [12 3 288 288]     for 4x4: [6, 48, 96, 96] --> [6, 3, 384, 384] 
            else:
                focal_x = rearrange(torch.cat(x[rand_views:]), 'b (th tw c) h w -> b c (th h) (tw w)', th=4, c=3) #[12 27 96 96] --> [12 3 288 288]     for 4x4: [6, 48, 96, 96] --> [6, 3, 384, 384
            # print("rand_x:",rand_x.shape)   #torch.Size([2, 3, 672, 672])    torch.Size([1, 3, 896, 896])
            # print("focal_x:",focal_x.shape) #torch.Size([12, 3, 288, 288])   torch.Size([6, 3, 384, 384])
            

            # self.save_image(rand_x,"rand")
            # self.save_image(focal_x,"focal")
            # exit(0)
            # x=[rand_x,focal_x]
            # print("x[0].shape:",x[0].shape)
            # print("x[1].shape:",x[1].shape)
            
            # for end_idx in idx_crops:
            
            # x = self.forward_features(x, msn_pretraining)
            
            h = self.forward_features(rand_x, msn_pretraining)
            # print("Inside forward rand H:",h.shape)     #torch.Size([2, 256])
            h = torch.cat((h, self.forward_features(focal_x, False)))
            # print("Inside forward rand + focal H:",h.shape) #torch.Size([14, 256])
            
            # print(self.head)
        
        #exit(0)

        
        
        
        
        return h

# DeiT III: Revenge of the ViT (https://arxiv.org/abs/2204.07118)

@register_model
def deit_tiny_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,   **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    
    return model
    
    
@register_model
def deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False, duration=8, super_img_rows=1, patch_drop=0.5, **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, duration=duration, super_img_rows=super_img_rows, patch_drop=patch_drop, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_small_'+str(img_size)+'_'
        if pretrained_21k:
            name+='21k.pth'
            print("Loading IN-21k ckpt!!!")
        else:
            name+='1k.pth'
            print("Loading IN-1k ckpt!!!")
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        # checkpoint["model"].pop("pos_embed", None)
        checkpoint["model"].pop("head.weight", None)
        checkpoint["model"].pop("head.bias", None)
        model.load_state_dict(checkpoint["model"], strict=False)

    return model

@register_model
def deit_medium_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False, **kwargs):
    model = vit_models(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers = Layer_scale_init_Block, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_medium_'+str(img_size)+'_'
        if pretrained_21k:
            name+='21k.pth'
        else:
            name+='1k.pth'
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        # model.load_state_dict(checkpoint["model"])
        checkpoint["model"].pop("head.weight", None)
        checkpoint["model"].pop("head.bias", None)
        model.load_state_dict(checkpoint["model"], strict=False)
    return model 

@register_model
def deit_base_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_base_'+str(img_size)+'_'
        if pretrained_21k:
            name+='21k.pth'
        else:
            name+='1k.pth'
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        # model.load_state_dict(checkpoint["model"])
        checkpoint["model"].pop("head.weight", None)
        checkpoint["model"].pop("head.bias", None)
        model.load_state_dict(checkpoint["model"], strict=False)
    return model
    
@register_model
def deit_large_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_large_'+str(img_size)+'_'
        if pretrained_21k:
            name+='21k.pth'
        else:
            name+='1k.pth'
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        # model.load_state_dict(checkpoint["model"])
        checkpoint["model"].pop("head.weight", None)
        checkpoint["model"].pop("head.bias", None)
        model.load_state_dict(checkpoint["model"], strict=False)
    return model
    
@register_model
def deit_huge_patch14_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers = Layer_scale_init_Block, **kwargs)
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_huge_'+str(img_size)+'_'
        if pretrained_21k:
            name+='21k_v1.pth'
        else:
            name+='1k_v1.pth'
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        # model.load_state_dict(checkpoint["model"])
        checkpoint["model"].pop("head.weight", None)
        checkpoint["model"].pop("head.bias", None)
        model.load_state_dict(checkpoint["model"], strict=False)
    return model
    
@register_model
def deit_huge_patch14_52_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=14, embed_dim=1280, depth=52, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers = Layer_scale_init_Block, **kwargs)

    return model
    
@register_model
def deit_huge_patch14_26x2_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=14, embed_dim=1280, depth=26, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers = Layer_scale_init_Block_paralx2, **kwargs)

    return model
    
@register_model
def deit_Giant_48x2_patch14_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=14, embed_dim=1664, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers = Block_paral_LS, **kwargs)

    return model

@register_model
def deit_giant_40x2_patch14_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers = Block_paral_LS, **kwargs)
    return model

@register_model
def deit_Giant_48_patch14_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=14, embed_dim=1664, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers = Layer_scale_init_Block, **kwargs)
    return model

@register_model
def deit_giant_40_patch14_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers = Layer_scale_init_Block, **kwargs)
    #model.default_cfg = _cfg()

    return model

# Models from Three things everyone should know about Vision Transformers (https://arxiv.org/pdf/2203.09795.pdf)

@register_model
def deit_small_patch16_36_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=36, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)

    return model
    
@register_model
def deit_small_patch16_36(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=36, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model
    
@register_model
def deit_small_patch16_18x2_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=18, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block_paralx2, **kwargs)

    return model
    
@register_model
def deit_small_patch16_18x2(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=18, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Block_paralx2, **kwargs)

    return model
    
  
@register_model
def deit_base_patch16_18x2_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=18, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block_paralx2, **kwargs)

    return model


@register_model
def deit_base_patch16_18x2(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=18, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Block_paralx2, **kwargs)

    return model
    

@register_model
def deit_base_patch16_36x1_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=36, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)

    return model

@register_model
def deit_base_patch16_36x1(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=36, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model
