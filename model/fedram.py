""" Federated Text-driven Prompt Generation for Vision-Language Models (ICLR 2024).
Copyright (c) 2024 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import pdb
import torch
import torch.nn as nn
from convclip import clip
from convclip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import numpy as np
_tokenizer = _Tokenizer()
import torch.nn.functional as F
CUSTOM_TEMPLATES =  "a photo of a {}." 
 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import clip 
from model.ram.base import BaseModel
from model.ram.ot_solver import Sinkhorn



def mmc_loss(logits, mask, logits_mask=None):
    """
        MMC Loss (Multi-Matching Contrastive loss)
        Inspired by SupCon Loss: https://github.com/google-research/google-research/tree/master/supcon
        MMC extends it into (1) multi-modal setting and (2) batched contrastive process
        Args:
            logits: torch.Tensor[B, C], B - mini-batch size, C - number of classes
            mask: torch.Tensor[B, C], binary mask, 1 if the class is present in the image, 0 otherwise
            logits_mask: torch.Tensor[B, C], mask out self-matching logits, not applied in multi-modal setting
        Returns:
            loss_cl: torch.Tensor[1], mean cross-entropy loss over positive pairs
    """
    # flatten the batch dimension
    logits = logits.reshape(-1)
    mask = mask.reshape(-1)
    
    # for numerical stability
    logits_max = torch.max(logits)
    logits = logits - logits_max.detach()
    exp_mixed_logits = torch.exp(logits)

    # mask out self-matching logits
    if logits_mask is not None:
        logits_mask = logits_mask.reshape(-1)
        exp_mixed_logits = exp_mixed_logits * logits_mask

    # cross entropy + softmax
    log_prob = logits - torch.log(exp_mixed_logits.sum())  
    num_pos_pairs = mask.sum()

    # sum over positive pairs, division is outside the log
    num_pos_pairs = torch.where(num_pos_pairs < 1e-6, 1, num_pos_pairs)
    mean_log_prob_pos = (mask * log_prob).sum() / num_pos_pairs    
    
    # mean over batch samples
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

class FedRAMCLIP(BaseModel):
    def __init__(
            self,
            cfg,
            clip_model,
            classnames_seen,
            classnames_unseen,
            clip_model_teacher,
            device=None
    ):
        super().__init__()
        self.cfg = cfg
        self.classnames_seen = classnames_seen
        self.classnames_unseen = classnames_unseen
        self.num_classes_seen = len(classnames_seen)
        self.num_classes_unseen = len(classnames_unseen)
        self.criterion = mmc_loss
        self.device = dist.get_rank() if cfg.TRAINER.RAM.DIST_TRAIN else 'cuda'

        # create teacher model, preserve teacher text features
        self.clip_model = clip_model
        self.text_tokens = self.get_text_templates()
        clip_model_teacher = clip_model_teacher.cuda()
        self.text_fea_teacher = self.get_text_fea(clip_model_teacher, classnames_seen+classnames_unseen)
        self.clip_model_teacher = clip_model_teacher

        # freeze the model
        self.freeze(cfg.TRAINER.RAM.TRANSFER_TYPE)

        # pre-trained logit scale
        self.logit_scale = self.clip_model.logit_scale.exp()
        # learnable temperature
        self.temperature_loc = nn.Parameter(torch.tensor(cfg.TRAINER.RAM.TEMPERATURE))
        self.temperature = nn.Parameter(1./self.logit_scale)
        
        # KCOT parameters
        self.reg = cfg.TRAINER.RAM.OT_REG
        self.reg_sc = cfg.TRAINER.RAM.OT_REGSC

    def get_text_fea(self, clip_model, classnames):
        text_templates = "A photo of a {}."
        text_templates = [text_templates.format(classnames[i]) for i in range(len(classnames))]
        text_tok = clip.tokenize(text_templates).cuda()
        with torch.no_grad():
            text_fea = clip_model.encode_text(text_tok)
        return text_fea.unsqueeze(1).detach()
    
    def get_text_templates(self):
        templates = "A photo of a {}."
        texts = [templates.format(name) for name in self.classnames_seen+self.classnames_unseen]
        text_tokens = clip.tokenize(texts)
        return text_tokens.cuda()
    
    def build_weights(self, sim, dim=-1, temperature=0.1):
        with torch.no_grad():
            sim_max = sim.max(dim=dim)[0]
            weights = (sim_max / temperature).softmax(dim=-1)
        return weights

    def generate_teacher_distribution(self, img_teacher, zsl=False, gzsl=False):
        with torch.no_grad():
            _, img_loc = self.clip_model_teacher.visual(img_teacher)
            img_loc = img_loc[0][:, 1:]
            text_fea = self.text_fea_teacher.clone().cuda()
            if zsl:
                text_fea = text_fea[self.num_classes_seen:]
            elif gzsl:
                pass
            else:
                text_fea = text_fea[:self.num_classes_seen]
            B, tok, dim=img_loc.shape
            C, gp, dim = text_fea.shape

            text_fea = F.normalize(text_fea, dim=-1) 
            img_loc = F.normalize(img_loc, dim=-1)
            text_fea = text_fea.unsqueeze(0).expand(B, -1, -1, -1).reshape(B, -1, dim)

            # generate weight
            logit_scale = self.clip_model_teacher.logit_scale.exp()
            logits_loc = logit_scale * img_loc @ text_fea.transpose(-2, -1) 
            logits_loc = logits_loc.reshape(B, -1, C, gp)
            local_similarity = logits_loc.softmax(dim=2)
            prob = (local_similarity*20.).softmax(dim=1)
            prob = prob.mean(dim=-1)
        return prob

    def forward(self, img, target=None, zsl=False, gzsl=True,req=False):
        seen = True if not zsl and not gzsl else False
        if seen:
            text_tokens = self.text_tokens[:self.num_classes_seen].clone()
        elif zsl:
            text_tokens = self.text_tokens[self.num_classes_seen:self.num_classes_seen+self.num_classes_unseen].clone()
        else:
            text_tokens = self.text_tokens[:self.num_classes_seen+self.num_classes_unseen].clone()
        prompt_fea_loc = self.clip_model.encode_text(text_tokens)
        prompt_fea_loc = prompt_fea_loc.unsqueeze(1)
        
        img_glb, img_loc = self.clip_model.visual(img, text_fea=prompt_fea_loc)
        img_loc = img_loc[0][:, 1:]

        B, tok, dim = img_loc.shape
        C, gp, dim = prompt_fea_loc.shape

        prompt_fea_loc = prompt_fea_loc.permute(1, 0, 2)

        img_glb = F.normalize(img_glb, dim=-1)
        img_loc = F.normalize(img_loc, dim=-1)
        prompt_fea_loc = F.normalize(prompt_fea_loc, dim=-1)

        logits_glb = img_glb @ prompt_fea_loc.transpose(1, 2) / self.temperature
        score_glb = logits_glb.squeeze(1).softmax(dim=-1)
        if req:
            mask = target
            loss_glb = self.criterion(logits_glb, mask)
        
        # Cost matrix
        sim = img_loc @ prompt_fea_loc.transpose(1, 2)
        cost = (sim * self.logit_scale).softmax(dim=-1)
        cost = 1.0 - cost

        if req:
            # Teacher is only applied in training
            frozen_mask = self.generate_teacher_distribution(img, zsl, gzsl)
            gt_mask = target.unsqueeze(1).expand(-1, tok, -1)
            frozen_mask[gt_mask==0] = frozen_mask.min()
            cost_tr = -torch.log(frozen_mask) * self.reg_sc
            cost = cost + cost_tr
            reg = self.reg + self.reg_sc
        else:
            reg = self.reg

        u = self.build_weights(sim.detach(), dim=2, temperature=0.1)
        v = torch.zeros((B, C), dtype=sim.dtype, device=sim.device).fill_(1./C)
        with torch.no_grad():
            T = Sinkhorn(u, v, cost, reg=reg)
        if torch.isnan(T).any():
            raise ValueError("Found nan in OT matrix!")
        
        sim_op = T * sim
        sim_op = sim_op.sum(dim=1) / self.temperature_loc
        score_loc = sim_op.softmax(dim=-1)
        score = (score_glb + score_loc) / 2.
        if req:
            mask = target
            loss_loc = self.criterion(sim_op, mask)
            loss = loss_glb + loss_loc
            return score, loss 
        else:
            return score




 