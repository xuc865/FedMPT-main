import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from collections import OrderedDict
import scipy.io as sio


from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from convclip import clip
from convclip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import tqdm
_tokenizer = _Tokenizer()
import numpy as np
import copy


CUSTOM_TEMPLATES_ori = "a photo of a {}." 

CUSTOM_TEMPLATES = {
    "OxfordPets": "X X X X {}, a type of pet.",
    "OxfordFlowers": "X X X X {}, a type of flower.",
    "FGVCAircraft": "X X X X {}, a type of aircraft.",
    "DescribableTextures": "X X X X {} texture.",
    "EuroSAT": "X X X X {}.",
    "StanfordCars": "X X X X {}, a type of car",
    "Food101": "X X X X {}, a type of food.",
    "SUN397": "X X X X {}.",
    "Caltech101": "X X X X {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}





class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, class_feature, weight, tokenized_prompts,flag=False):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if flag:
            x = self.transformer(x)
        else:
            counter=0
            outputs = self.transformer.resblocks([x,class_feature,weight,counter])
            x = outputs[0]

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            print("use given words to initialize context vectors")
            temp = 'a photo of a'
            ctx_init = temp.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

            ctx_vectors_src = embedding[0, 1 : 1 + n_ctx, :]

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)


        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        clip_model_ = clip_model
        clip_model_.cuda()
        clip_model = clip_model.cuda()
        temp = CUSTOM_TEMPLATES_ori 
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.cuda()

        with torch.no_grad():
            text_features = clip_model_.encode_text(prompts_)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        vis_dim = clip_model.visual.output_dim
        self.meta_net = nn.Sequential(
            OrderedDict([("linear1", nn.Linear(vis_dim, vis_dim // 4,bias=True)),
                         ("relu", QuickGELU()),
                         ("linear2", nn.Linear(vis_dim // 4, 4*ctx_dim,bias=True))
                         ]))
        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()
        classnames = [name.replace("_", " ") for name in classnames]
        temp = CUSTOM_TEMPLATES_ori
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(prompts)

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        self.prev_ctx=None

    def forward(self):
        class_feature = self.meta_net(self.text_features)
        class_feature = class_feature.reshape(class_feature.shape[0],-1,512)
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx
        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prompt = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompt, class_feature


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class TCPCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.ori_embedding = self.prompt_learner.text_features
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.domain_sim = -1
        self.domain_sim_src = -1
        self.weight = cfg.TRAINER.COOP.W
    
    def forward(self, image, label=None, req=False):
        image_features = self.image_encoder(image.type(self.dtype))
        text_features_old = self.ori_embedding
        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
        text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts,class_prompt = self.prompt_learner()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = self.text_encoder(prompts, class_prompt, self.weight, tokenized_prompts.detach()) 
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # logits = logit_scale.detach() * image_features.detach() @ text_features_norm.t()
       
        # Class-Specific Region Feature Aggregation
        output = 20 * F.conv1d(image_features, text_features[:, :, None])
        b, c, _ = output.shape
        output_half = output
        w_half = F.softmax(output_half, dim=-1)
        w = w_half
        output = 5 * (output * w).sum(-1)
        b, c = output.shape
        logits = output
          
        if req:
            score= cos(text_features,text_features_old)
            score  = 1.0-torch.mean(score)
            loss = 8.0*score
            return logits, loss
          
        return logits

