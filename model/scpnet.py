from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
import pdb
from convclip import clip
from convclip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import os
from omegaconf import OmegaConf
 
pref = "PATHPDBB/FedMPT-main/model/scps/"
# Base config
def get_config(name):
    base_cfg = OmegaConf.load(pref+'base.yaml')
    if name == "voc":
        main_cfg = OmegaConf.load(pref+'scpnet+voc.yaml')
    elif name == "coco":
        main_cfg = OmegaConf.load(pref+'scpnet+coco.yaml')
    elif name == "nus":
        main_cfg = OmegaConf.load(pref+'scpnet+nuswide.yaml')
    elif name == "multiscene":
        main_cfg = OmegaConf.load(pref+'scpnet+multiscene.yaml')
    elif name == "mlrsnet":
        main_cfg = OmegaConf.load(pref+'scpnet+mlrsnet.yaml')
    else:
        raise NotImplementedError
    cfg = OmegaConf.merge(base_cfg, main_cfg)
    return cfg
 
_tokenizer = _Tokenizer()
  

class TextEncoder(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]),
              tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.n_ctx
        dtype = clip_model.dtype
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # use given words to initialize context vectors
        ctx_init = cfg.ctx_init.replace("_", " ")
        assert (n_ctx == len(ctx_init.split(" ")))
        prompt = clip.tokenize(ctx_init)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1:1 + n_ctx, :]
        prompt_prefix = ctx_init

        self.ctx = nn.Parameter(ctx_vectors)  # type: ignore
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(
                dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix",
                             embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.register_buffer("token_middle", embedding[:, 1:(1 + n_ctx), :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],  # type: ignore
            dim=1,
        )
        return prompts


import math
import numpy as np
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.parameter.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

from timm.models.vision_transformer import resize_pos_embed
class SCPNet(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        cfg = get_config(cfg.DATASET.NAME)
        self.cfg = cfg
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.gc1 = GraphConvolution(1024//2, 2048//2).to(self.dtype)
        self.gc2 = GraphConvolution(2048//2, 2048//2).to(self.dtype)
        self.gc3 = GraphConvolution(2048//2, 1024//2).to(self.dtype)
        self.relu = nn.LeakyReLU(0.2)
        self.relu2 = nn.LeakyReLU(0.2)
       
        self.relation = torch.Tensor(np.load(cfg.relation_file)).to(self.dtype)
        
        _ ,max_idx = torch.topk(self.relation, cfg.sparse_topk)
        mask = torch.ones_like(self.relation).type(torch.bool)
        for i, idx in enumerate(max_idx):
            mask[i][idx] = 0
        self.relation[mask] = 0
        sparse_mask = mask
        dialog = torch.eye(cfg.num_classes).type(torch.bool)
        self.relation[dialog] = 0
        self.relation = self.relation / torch.sum(self.relation, dim=1).reshape(-1, 1) * cfg.reweight_p
        self.relation[dialog] = 1-cfg.reweight_p

        self.gcn_relation = self.relation.clone()
        assert(self.gcn_relation.requires_grad == False)
        self.relation = torch.exp(self.relation/cfg.T) / torch.sum(torch.exp(self.relation/cfg.T), dim=1).reshape(-1,1)
        self.relation[sparse_mask] = 0
        self.relation = self.relation / torch.sum(self.relation, dim=1).reshape(-1, 1)

    def forward(self, image):
        tokenized_prompts = self.tokenized_prompts
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        logit_scale = self.logit_scale.exp()
        if self.cfg.scale != 'clip':
            assert(isinstance(self.cfg.scale, int))
            logit_scale = self.cfg.scale
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        identity = text_features

        text_features = self.gc1(text_features, self.gcn_relation.cuda())
        text_features = self.relu(text_features)
        text_features = self.gc2(text_features, self.gcn_relation.cuda())
        text_features = self.relu2(text_features)
        text_features = self.gc3(text_features, self.gcn_relation.cuda())
        
        text_features += identity
        text_features = text_features / text_features.norm(dim=-1,
                                                            keepdim=True)
        # logits = logit_scale * image_features @ text_features.t()
        # return logits
        # Class-Specific Region Feature Aggregation
        output = 20 * F.conv1d(image_features, text_features[:, :, None])
        b, c, _ = output.shape
        output_half = output
        w_half = F.softmax(output_half, dim=-1)
        w = w_half
        output = 5 * (output * w).sum(-1)
        b, c = output.shape
        logits = output
        return logits
