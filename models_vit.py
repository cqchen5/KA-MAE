# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed, Block
from util.patch_embed import PatchEmbed_new, PatchEmbed_org, PatchEmbed_SSL

from transformers import Wav2Vec2FeatureExtractor, HubertModel, Wav2Vec2Model, WavLMModel

class MLPLayer(nn.Module):
    def __init__(self, dim_input, num_class=6):
        super(MLPLayer, self).__init__()
        self.dnn = nn.Linear(dim_input, dim_input)
        self.relu = nn.ReLU()
        self.classification = nn.Linear(dim_input, num_class)
        # self.dropout = nn.Dropout(0.)

        nn.init.xavier_uniform_(self.dnn.weight.data)
        nn.init.zeros_(self.dnn.bias.data)
        nn.init.xavier_uniform_(self.classification.weight.data)
        nn.init.zeros_(self.classification.bias.data)

    def forward(self, x):
        # x = self.dropout(x)
        x = self.dnn(x)
        x = self.relu(x)
        ctc_out = self.classification(x)
        return ctc_out


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, mask_2d=False, use_custom_patch=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        embed_dim = kwargs['embed_dim']
        norm_layer = kwargs['norm_layer']
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm
        else:
            self.norm = norm_layer(embed_dim)

        self.mask_2d = mask_2d
        self.use_custom_patch = use_custom_patch
        num_heads=12
        depth=8
        mlp_ratio=4

        self.patch_embed = PatchEmbed_org((1024, 256), 16, 1, embed_dim)
        self.patch_embed_ssl = PatchEmbed_SSL((512, 768), (8, 48), 1, embed_dim)
        num_patches = self.patch_embed.num_patches
        num_patches_ssl = self.patch_embed_ssl.num_patches
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.pos_embed_ssl = nn.Parameter(torch.zeros(1, num_patches_ssl + 1, embed_dim), requires_grad=False) 
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_ssl = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # breakpoint()
        # self.pos_drop = nn.Dropout(p=0.)

        self.blocks_Mag = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(4)])
        
        self.blocks1 = nn.ModuleList([
            Block(embed_dim*2, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(int(depth/2))])
        
        self.dowmsample = nn.Linear(embed_dim*2, embed_dim, bias=True)

        self.blocks2 = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(int(depth/2))])

        self.num_layers = 8
        
        # 初始化为全 0，经过 Softmax 后初始权重均为 1/25
        self.layer_weights = nn.Parameter(torch.zeros(self.num_layers))
        
        # del self.blocks
        

        self.emo2vec = AutoModel(model="/raw22/asrprg/permanent/glzhong/audio_understand_LLM/1.models/emotion2vec/emotion2vec", disable_update=True)
        self.emo2vec.model.eval()
        for p in self.emo2vec.model.parameters():
            p.requires_grad = False
        model_name = "/train20/sppro/permanent/cqchen5/model/WavLM_Large"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name)
        self.model.eval()
        

        self.num_classes = kwargs['num_classes']

        # self.mlp_layer1 = MLPLayer(embed_dim, 256)
        self.head = nn.Linear(embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()  #trunc_normal_(model.head.weight, std=2e-5) 初始化


    def forward_features(self, x, audio):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)    

        # print("audio.shape: ", audio.shape)
        with torch.no_grad():
            # breakpoint()
            audio_list = audio.tolist()
            input_values = self.feature_extractor(audio_list, sampling_rate=16000, padding=True, return_tensors="pt").input_values
            ssl = self.model(input_values.cuda(), output_hidden_states=True)
        
        layer_reps = ssl.hidden_states[1:]
        stacked_states = torch.stack(layer_reps, dim=1)
        norm_weights = F.softmax(self.layer_weights, dim=-1).cuda()
        reshaped_weights = norm_weights.view(1, self.num_layers, 1, 1)
        weighted_output = (stacked_states * reshaped_weights).sum(dim=1)
        zeros = torch.zeros(B, 1, 1024, device=weighted_output.device, dtype=weighted_output.dtype)
        x_ssl = torch.cat([weighted_output, zeros], dim=1)
        
        x_ssl = self.patch_embed_ssl(x_ssl)
        x_ssl = x_ssl + self.pos_embed_ssl[:, 1:, :]
        cls_token_ssl = self.cls_token_ssl + self.pos_embed_ssl[:, :1, :]
        cls_token_ssls = cls_token_ssl.expand(x_ssl.shape[0], -1, -1)
        x_ssl = torch.cat((cls_token_ssls, x_ssl), dim=1)
        x_ssl = self.pos_drop(x_ssl)     

        for blk in self.blocks_Mag:
            x = blk(x)
            # breakpoint()
        
        x_concat = torch.cat([x, x_ssl], dim=2)
        # weights = F.softmax(self.logits, dim=0)
        # x_concat = weights[0] * x + weights[1] * x_ssl
        
        # apply Transformer blocks
        for blk in self.blocks1:
            x_concat = blk(x_concat)

        x_concat = self.dowmsample(x_concat)

        for blk in self.blocks2:
            x_concat = blk(x_concat)

        if self.global_pool:
            # x_concat = self.mlp_layer1(x_concat)
            x_concat = x_concat[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x_concat)
        else:
            x_concat = self.norm(x_concat)
            outcome = x_concat[:, 0]

        return outcome

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def random_masking_2d(self, x, x_ssl, mask_t_prob, mask_f_prob):
        """
        2D: Spectrogram (msking t and f under mask_t_prob and mask_f_prob)
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        
        N, L, D = x.shape  # batch, length, dim
        if self.use_custom_patch:
            # # for AS
            T=101 #64,101
            F=12 #8,12
            # # for ESC
            # T=50
            # F=12 
            # for SPC
            # T=12
            # F=12
        else:
            # ## for AS 
            T=64
            F=16
            # ## for ESC
            #T=32
            #F=8            
            ## for SPC
            # T=8
            # F=8
        
        # mask T
        x = x.reshape(N, T, F, D)
        x_ssl = x_ssl.reshape(N, T, F, D)
        len_keep_T = int(T * (1 - mask_t_prob))
        noise = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_T]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, F, D)
        #x_masked = torch.gather(x, dim=1, index=index)
        #x_masked = x_masked.reshape(N,len_keep_T*F,D)
        x = torch.gather(x, dim=1, index=index) # N, len_keep_T(T'), F, D
        x_ssl = torch.gather(x_ssl, dim=1, index=index) # N, len_keep_T(T'), F, D

        # mask F
        #x = x.reshape(N, T, F, D)
        x = x.permute(0,2,1,3) # N T' F D => N F T' D
        x_ssl = x_ssl.permute(0,2,1,3) # N T' F D => N F T' D
        len_keep_F = int(F * (1 - mask_f_prob))
        noise = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_F]
        #index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, T, D)
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, len_keep_T, D)
        x_masked = torch.gather(x, dim=1, index=index)
        x_masked = x_masked.permute(0,2,1,3) # N F' T' D => N T' F' D 
        #x_masked = x_masked.reshape(N,len_keep*T,D)
        x_masked = x_masked.reshape(N,len_keep_F*len_keep_T,D)

        x_ssl_masked = torch.gather(x_ssl, dim=1, index=index)
        x_ssl_masked = x_ssl_masked.permute(0,2,1,3) # N F' T' D => N T' F' D 
        #x_masked = x_masked.reshape(N,len_keep*T,D)
        x_ssl_masked = x_ssl_masked.reshape(N,len_keep_F*len_keep_T,D)
            
        return x_masked, x_ssl_masked, None, None, index


    def forward_features_mask(self, x, audio, mask_t_prob, mask_f_prob):
        B = x.shape[0] #4,1,1024,128

        with torch.no_grad():
            # breakpoint()
            audio_list = audio.tolist()
            input_values = self.feature_extractor(audio_list, sampling_rate=16000, padding=True, return_tensors="pt").input_values
            ssl = self.model(input_values.cuda(), output_hidden_states=True)
        
        layer_reps = ssl.hidden_states[1:]
        stacked_states = torch.stack(layer_reps, dim=1)
        norm_weights = F.softmax(self.layer_weights, dim=-1).cuda()
        reshaped_weights = norm_weights.view(1, self.num_layers, 1, 1)
        weighted_output = (stacked_states * reshaped_weights).sum(dim=1)
        zeros = torch.zeros(B, 1, 1024, device=weighted_output.device, dtype=weighted_output.dtype)
        x_ssl = torch.cat([weighted_output, zeros], dim=1)

        # embed patches
        # breakpoint()
        x = self.patch_embed(x)
        x_ssl = self.patch_embed_ssl(x_ssl)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        x_ssl = x_ssl + self.pos_embed_ssl[:, 1:, :]

        if self.mask_2d:
            x, x_ssl, mask, ids_restore, ids_keep = self.random_masking_2d(x, x_ssl, mask_t_prob, mask_f_prob)
            # _, _, D = x_ssl.shape
            # breakpoint()
            # x_ssl = torch.gather(x_ssl, dim=1, index=ids_keep)  # 有问题
            # x_ssl = torch.gather(x_ssl, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        else:
            # breakpoint()
            x, mask, ids_restore, ids_keep = self.random_masking(x, mask_t_prob)
            _, _, D = x_ssl.shape
            x_ssl = torch.gather(x_ssl, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)    

        cls_token_ssl = self.cls_token_ssl + self.pos_embed_ssl[:, :1, :]
        cls_token_ssls = cls_token_ssl.expand(x_ssl.shape[0], -1, -1)
        x_ssl = torch.cat((cls_token_ssls, x_ssl), dim=1)

        x = self.pos_drop(x)
        x_ssl = self.pos_drop(x_ssl)

        for blk in self.blocks_Mag:
            x = blk(x)
            # breakpoint()
        
        x_concat = torch.cat([x, x_ssl], dim=2)
        # weights = F.softmax(self.logits, dim=0)
        # x_concat = weights[0] * x + weights[1] * x_ssl
        
        # apply Transformer blocks
        for blk in self.blocks1:
            x_concat = blk(x_concat)

        x_concat = self.dowmsample(x_concat)

        for blk in self.blocks2:
            x_concat = blk(x_concat)

        if self.global_pool:
            # x_concat = self.mlp_layer1(x_concat)
            x_concat = x_concat[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x_concat)
        else:
            x_concat = self.norm(x_concat)
            outcome = x_concat[:, 0]

        return outcome



    # overwrite original timm
    def forward(self, x, audio, v=None, mask_t_prob=0.0, mask_f_prob=0.0):
        # breakpoint()
        if mask_t_prob > 0.0 or mask_f_prob > 0.0:
            x = self.forward_features_mask(x, audio, mask_t_prob=mask_t_prob, mask_f_prob=mask_f_prob)
        else:
            x = self.forward_features(x, audio)
        # print("Before", x.shape)
        x = self.head(x)
        # print("After", x.shape)
        return x



def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)        
    return model

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=8, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
