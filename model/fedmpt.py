"""
FedMPT (CVPR 2026): Federated Multi-Label Prompt Tuning.

Core idea (see paper §4):
  - Multiple LLM-derived *conditions* (e.g. scene layout) → condition prompts
  - Patch–text alignment + per-condition aggregation (gating over conditions)
  - Only prompt_learner weights are trained and federated via FedAvg
"""
import torch
import torch.nn as nn
from convclip import clip
from convclip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch.nn.functional as F
from model.ram.ot_solver import Sinkhorn

_tokenizer = _Tokenizer()

# Abstract conditions from the paper's LLM pipeline (Fig. 3); use --cond to set how many are active
DEFAULT_CONDITIONS = [
    "color",
    "background",
    "scene type",
    "spatial layout",
    "pose",
    "lighting",
    "object scale",
]


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
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class CoOpPromptLearner(nn.Module):
    """Learnable class/condition tokens, visual adapter, and condition router (paper §4.3)."""

    def __init__(self, cfg, dtype):
        super().__init__()
        n_ctx = cfg.TRAINER.ML.CLS
        ctx_vectors = torch.empty(n_ctx, 512)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        n_attr_ctx = cfg.TRAINER.ML.COND
        attr_vectors = torch.empty(n_attr_ctx, 512)
        nn.init.normal_(attr_vectors, std=0.02)
        self.attr_ctx = nn.Parameter(attr_vectors)

        self.cross_attention = nn.MultiheadAttention(512, 8, 0.1).to(dtype)
        latentdim = cfg.TRAINER.ML.LAT
        self.pro = nn.Sequential(
            nn.Linear(512, latentdim),
            nn.ReLU(),
            nn.Linear(latentdim, 768),
        ).to(dtype)
        # Router Ω: global patch feature -> per-condition weights (Eq. 11)
        self.router = nn.Sequential(
            nn.Linear(512, latentdim),
            nn.ReLU(),
            nn.Linear(latentdim, n_attr_ctx),
        ).to(dtype)

    def forward(self):
        return self.ctx, self.attr_ctx


class FedMPTCLIP(nn.Module):
    def __init__(self, cfg, clip_model, classnames, device="cuda"):
        super().__init__()
        self.cfg = cfg
        self.set_prompt_prefix()

        self.prompt_learner = CoOpPromptLearner(cfg, clip_model.dtype)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.token_embedding = clip_model.token_embedding
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = device
        self.clip_model_ = clip_model
        self.classnames = classnames

    def set_prompt_prefix(self):
        n_ctx = self.cfg.TRAINER.ML.CLS
        self.n_ctx = n_ctx
        self.prompt_prefix = " ".join(["X"] * n_ctx)
        self.prefix1 = " ".join(["X"] * self.cfg.TRAINER.ML.COND)

        n_cond = self.cfg.TRAINER.ML.COND
        if n_cond > len(DEFAULT_CONDITIONS):
            raise ValueError(
                f"--cond={n_cond} exceeds built-in condition list "
                f"({len(DEFAULT_CONDITIONS)}). Extend DEFAULT_CONDITIONS in "
                "model/fedmpt.py or lower --cond."
            )
        self.attn_text = DEFAULT_CONDITIONS[:n_cond]
        self.num_cond = n_cond
        self.agg_mode = getattr(self.cfg.TRAINER.ML, "AGG", "vanilla")
        self.ot_reg = float(getattr(self.cfg.TRAINER.ML, "OT_REG", 0.1))
        print(
            f"FedMPT: {self.num_cond} conditions -> {self.attn_text}, "
            f"agg={self.agg_mode}"
        )

    @staticmethod
    def _patch_importance(sim, temperature=0.1):
        """Marginal over patches (Fed-RAM build_weights, paper Eq. 8)."""
        with torch.no_grad():
            sim_max = sim.max(dim=-1)[0]
            return (sim_max / temperature).softmax(dim=-1)

    def _ot_class_scores(self, image_features, text_n):
        """
        Sinkhorn OT between patches and classes for one condition (paper §4.2).
        image_features: (B, D, L), text_n: (C, D) -> scores (B, C).
        """
        sim = torch.einsum("bdl,cd->blc", image_features, text_n)
        # Fed-RAM: cost = 1 - softmax(sim * scale); vanilla path uses 20× conv1d
        cost = 1.0 - (20.0 * sim).softmax(dim=-1)
        u = self._patch_importance(sim.detach())
        b, c = sim.shape[0], sim.shape[-1]
        v = torch.full((b, c), 1.0 / c, device=sim.device, dtype=sim.dtype)
        with torch.no_grad():
            transport = Sinkhorn(u, v, cost, reg=self.ot_reg)
        if torch.isnan(transport).any():
            raise ValueError("NaN in Sinkhorn transport plan.")
        return (transport * sim).sum(dim=1)

    def _vanilla_class_scores(self, image_features, text_n, temp):
        """Original conv1d + patch softmax aggregation for one condition."""
        weight = text_n.unsqueeze(-1).unsqueeze(0)  # (1, C, D, 1)
        output = 20 * F.conv1d(image_features, weight)
        w = F.softmax(output / temp, dim=-1)
        return (output.shape[-1]) * (output * w).sum(-1)

    def _fuse_conditions(self, Pn, image_features, temp):
        """Router over per-condition predictions (paper §4.3)."""
        gfeat = image_features.mean(dim=-1)
        omega = self.prompt_learner.router(gfeat.to(self.dtype))
        gate = F.softmax(omega / temp, dim=1)
        return (gate.unsqueeze(-1) * Pn).sum(dim=1)

    def forward(self, image, classnames=None, dataname=None, req=False):
        classnames = [name.replace("_", " ") for name in self.classnames]
        num_classes = len(classnames)

        text_features = self.encode_condition_text(classnames)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype), ori=True)
        prompt_embeddings = self.prompt_learner.pro(
            self.prompt_learner.cross_attention(
                image_features, text_features, text_features
            )[0]
        )
        image_features = self.image_encoder(
            image.type(self.dtype), prompt_embeddings.type(self.dtype)
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        temp = self.cfg.TRAINER.ML.TEMP
        pn_list = []
        for n in range(self.num_cond):
            text_n = text_features[n * num_classes : (n + 1) * num_classes]
            if self.agg_mode == "ot":
                pn_list.append(self._ot_class_scores(image_features, text_n))
            else:
                pn_list.append(self._vanilla_class_scores(image_features, text_n, temp))
        Pn = torch.stack(pn_list, dim=1)
        logits = self._fuse_conditions(Pn, image_features, temp)

        if req:
            return logits, 0
        return logits

    def encode_image(self, image):
        return self.image_encoder(image.type(self.dtype))

    def encode_condition_text(self, classnames):
        """Abstract condition phrase + learnable COND/CLASS tokens (paper §4.1)."""
        num_classes = len(classnames)
        prompts = [
            [
                self.prefix1 + " " + attn + " " + self.prompt_prefix + " " + name + "."
                for name in classnames
            ]
            for attn in self.attn_text
        ]
        flat = [p for group in prompts for p in group]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in flat])
        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts.to(self.device)).type(self.dtype)

        cond_len = self.cfg.TRAINER.ML.COND
        token_prefix = embedding[:, :1, :]
        token_midfix = embedding[:, 1 + cond_len : 1 + cond_len + 1, :]
        token_suffix = embedding[:, 1 + cond_len + 1 + self.n_ctx :, :]

        ctx, attr_ctx = self.prompt_learner()
        ctx = ctx.to(self.dtype).unsqueeze(0).expand(token_prefix.shape[0], -1, -1)
        # One learnable condition token vector per (condition, class) row
        row_idx = torch.arange(token_prefix.shape[0], device=attr_ctx.device)
        cond_idx = row_idx // num_classes
        attr_rows = attr_ctx[cond_idx].unsqueeze(1)
        prompt_vectors = torch.cat(
            [token_prefix, attr_rows, token_midfix, ctx, token_suffix], dim=1
        )
        return self.text_encoder(prompt_vectors, tokenized_prompts)
