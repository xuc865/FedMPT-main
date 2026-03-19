import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import pdb
# from Dassl.dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.engine.trainer import TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from convclip import clip
from convclip.simple_tokenizer import SimpleTokenizer as _Tokenizer

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

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.FEDPGP.N_CTX
        ctx_init = cfg.TRAINER.FEDPGP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        bottleneck = cfg.TRAINER.FEDPGP.BOTTLENECK
        self.N = cfg.TRAINER.FEDPGP.N
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if False: # ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization 
            print("Initializing a generic context")
            U = torch.empty(self.N,n_ctx, bottleneck, dtype=dtype)
            V = torch.empty(self.N,bottleneck, ctx_dim, dtype=dtype)
            sigma = torch.empty(self.N,n_ctx, ctx_dim, dtype = dtype)
            # ctx_vectors = torch.matmul(U,V)

            # nn.init.normal_(ctx_vectors, std=0.02)
            nn.init.normal_(U, std=0.02)
            nn.init.normal_(V, std=0.02)
            nn.init.normal_(sigma, std=0.02)# define the prompt to be trained
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.U = nn.Parameter(U)
        self.V = nn.Parameter(V)
        self.sigma = nn.Parameter(sigma)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.repeat(self.N, 1)
        # tokenized_prompts3.view(3,100,77)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

            # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.register_buffer("embedding", embedding)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.FEDPGP.CLASS_TOKEN_POSITION

    def forward(self):

        # ctx = self.ctx
        U = self.U
        V = self.V
        UV = torch.matmul(U,V)
        sigma = self.sigma
        ctx = UV +self.sigma
        embedding = self.embedding

        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)

        ctx = ctx.permute(1, 0, 2, 3)
        ctx = ctx.contiguous().view(self.N * self.n_cls, self.n_ctx, ctx.shape[3])

        if UV.dim() == 3:
            UV = UV.unsqueeze(0).expand(self.n_cls, -1, -1, -1)

        UV = UV.permute(1, 0, 2, 3)
        UV = UV.contiguous().view(self.N * self.n_cls, self.n_ctx, UV.shape[3])

        if sigma.dim() == 3:
            sigma = sigma.unsqueeze(0).expand(self.n_cls, -1, -1, -1)

        sigma = sigma.permute(1, 0, 2, 3)
        sigma = sigma.contiguous().view(self.N * self.n_cls, self.n_ctx, sigma.shape[3])

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            prompts_sigma = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    sigma,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            prompts_UV = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    UV,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return  embedding,prompts_sigma,prompts_UV,prompts


class FedPGPCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, req=False):

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        embedding,prompts_sigma,prompts_UV,prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts

        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
         
        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ text_features.t()
        # Class-Specific Region Feature Aggregation
        output = 20 * F.conv1d(image_features, text_features[:, :, None])
        b, c, _ = output.shape
        output_half = output
        w_half = F.softmax(output_half, dim=-1)
        w = w_half
        output = 5 * (output * w).sum(-1)
        b, c = output.shape
        logits = output[:,:output.shape[-1]//2]

        if req:
            text_features_0 = self.text_encoder(embedding, tokenized_prompts)
            text_features_sigma = self.text_encoder(prompts_sigma, tokenized_prompts)
            text_features_UV = self.text_encoder(prompts_UV, tokenized_prompts)

            text_features_0 = text_features_0 / text_features_0.norm(dim=-1, keepdim=True)
            text_features_sigma = text_features_sigma / text_features_sigma.norm(dim=-1, keepdim=True)
            text_features_UV = text_features_UV / text_features_UV.norm(dim=-1, keepdim=True) 
 
            cos = torch.nn.CosineSimilarity(dim=-1)
            # text_features_0,text_features_sigma,text_features_UV,text_features,output = self.model(image)
            posi = cos(text_features_0,text_features_sigma)
            nega = cos(text_features_sigma,text_features)
 
            logits_new = torch.cat((posi.reshape(-1, 1), nega.reshape(-1, 1)), dim=1)
            logits_new /= self.cfg.TRAINER.FEDPGP.temp 
            return logits, logits_new[:logits_new.shape[0]//2].T 

        return logits

 