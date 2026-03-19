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
import time
from model import *
from dataloader.dm_federated import TrainDataManager
from federated.utils import *
import torch.nn.functional as F
from federated.base_trainer import TrainerBase
from dataloader.dm_federated import TestDataManager 
from utilss import *
from torch.cuda.amp import autocast
from mldata.cls_to_names import *
from convclip import clip
from model.ram.make_scheduler import OneCycleLR, CosineLRScheduler

def make_scheduler(cfg, optimizer, dataloader_len):
    lr_scheduler = cfg.TRAINER.RAM.SOLVER.LR_SCHEDULER
    lr_mult = cfg.TRAINER.RAM.SOLVER.IMS_PER_BATCH / 256
    num_epochs = cfg.OPTIM.MAX_EPOCH
    lr_min = 0.2 * cfg.TRAINER.RAM.SOLVER.BASE_LR
    warmup_lr_init = 1 * cfg.TRAINER.RAM.SOLVER.BASE_LR

    warmup_t = cfg.TRAINER.RAM.SOLVER.WARMUP_EPOCHS
    noise_range = None
    
    if lr_scheduler == 'onecycle':
        lr_scheduler = OneCycleLR(
            optimizer, 
            max_lr=[cfg.TRAINER.RAM.SOLVER.BASE_LR_CLIP * lr_mult, cfg.TRAINER.RAM.SOLVER.BASE_LR * lr_mult], 
            steps_per_epoch=dataloader_len, 
            epochs=cfg.TRAINER.RAM.SOLVER.MAX_EPOCHS, 
            pct_start=0.2
        )
    else:
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=lr_min,
            t_mul= 1.,
            decay_rate=0.1,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_t,
            cycle_limit=1,
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct= 0.67,
            noise_std= 1.,
            noise_seed=42,
        )

    return lr_scheduler


def make_optimizer(cfg, model, lr_mult=0.1):
    clip_params, sgd_params, other_params = [], [], []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        elif 'embeds' in pname: 
            sgd_params.append(p)
        elif pname.startswith('clip'):
            clip_params.append(p)
        else:
            other_params.append(p)

    # Optimizer1
    param_groups = [
        {'params': clip_params, 'lr': cfg.TRAINER.RAM.SOLVER.BASE_LR * lr_mult, 'weight_decay': cfg.TRAINER.RAM.SOLVER.WEIGHT_DECAY},
        {'params': other_params, 'lr': cfg.TRAINER.RAM.SOLVER.BASE_LR, 'weight_decay': cfg.TRAINER.RAM.SOLVER.WEIGHT_DECAY},
        {'params': sgd_params, 'lr': cfg.TRAINER.RAM.SOLVER.BASE_LR_SGD, 'weight_decay': cfg.TRAINER.RAM.SOLVER.WEIGHT_DECAY_SGD}
    ]
    if cfg.TRAINER.RAM.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = torch.optim.SGD(param_groups, momentum=cfg.TRAINER.RAM.SOLVER.MOMENTUM)
    elif cfg.TRAINER.RAM.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(param_groups)
    else:
        optimizer = getattr(torch.optim, cfg.TRAINER.RAM.SOLVER.OPTIMIZER_NAME)(param_groups, momentum=cfg.TRAINER.RAM.SOLVER.MOMENTUM)
 
    return optimizer 


class Client(TrainerBase):
    """A local client with frozen clip and FL meta_net and private training data"""
    def __init__(self, cfg, client_id,dataname,available_cls,clip_model,nobuild=False,zsmodel=None):
        super().__init__()
        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.client_id = client_id
        self.best_mAP = 0
        # self.id = -1
        self.cfg = cfg
        if not nobuild:
            self.build_data_loader(dataname,available_cls, clip_model)
        self.build_model(clip_model,zsmodel)
        if self.cfg.MODEL.NAME in ["dualcoop", "poscoop"]:
            self.criterion = AsymmetricLoss_Dual(cfg.TRAINER.ML.ASL_GAMMA_NEG, cfg.TRAINER.ML.ASL_GAMMA_POS)
        else:
            self.criterion = AsymmetricLoss_Pos(cfg.TRAINER.ML.ASL_GAMMA_NEG, cfg.TRAINER.ML.ASL_GAMMA_POS) 
            

    def build_data_loader(self,dataname,available_cls, clip_model):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = TrainDataManager(self.cfg, dataname, clip_model, available_cls) 
        self.train_loader = dm.train_loader
        self.available_classes = dm.available_classes
        self.data_name = dm.data_name 
        
        dm = TestDataManager(self.cfg, dataname,  clip_model, available_cls)
        self.test_loader = dm.test_loader


    def build_model(self,clip_model,zsmodel):
        cfg = self.cfg

        classsnames = -1
        dna = cfg.DATASET.NAME
        if dna == "voc":
            classsnames = voc2007_classes
        elif dna == "coco":
            classsnames = coco2014_classes
        elif dna == "nus":
            classsnames = nuswide_classes
        elif dna == "object":
            classsnames = object365_classes
        elif dna == "multiscene":
            classsnames = multiscene_classes
        elif dna == "mlrsnet":
            classsnames = mlrsnet_classes
        else:
            raise NotImplementedError

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        # clip_model = load_clip_to_cpu(cfg)
        self.model_name = cfg.MODEL.NAME
        print("Building custom CLIP")
        nam = cfg.MODEL.NAME
        if nam == 'fedtpg':
            self.model = FedTPG(cfg, clip_model, classnames=classsnames,device = self.device)
        elif nam == "dualcoop":
            self.model = DualCoop(cfg, classnames=classsnames, clip_model=clip_model)
        elif nam == "poscoop":
            self.model = PositiveCoop(cfg, classnames=classsnames, clip_model=clip_model)
        elif nam == "scpnet":
            self.model = SCPNet(cfg, classnames=classsnames, clip_model=clip_model)
        elif nam == 'coop':
            self.model = CoOpCLIP(cfg, clip_model,classnames=classsnames,device = self.device)
        elif nam == 'fedawa':
            self.model = FedAWACLIP(cfg, clip_model, classnames=classsnames,device = self.device)
        elif nam == 'maple':
            self.model = MapleCLIP(cfg, classsnames, clip_model)
        elif nam == 'fedmvp':
            self.model = FedMVPCLIP(cfg, clip_model, classnames=classsnames,device = self.device)
        elif nam == 'fedmpt':
            self.model = FedMPTCLIP(cfg, clip_model,classnames=classsnames,device = self.device)
        elif nam == 'tcp':
            self.model = TCPCLIP(cfg, classsnames, clip_model)
        elif nam == 'fedpgp':
            self.model = FedPGPCLIP(cfg, classsnames, clip_model)
        elif nam == 'vlp':
            self.model = VLPCLIP(cfg, clip_model,device = self.device)
        elif nam == 'fedram':
            self.model_teacher = zsmodel
            self.model = FedRAMCLIP(cfg, clip_model,classsnames,[],self.model_teacher, device = self.device)
            
        self.w = cfg.TRAIN.W

        if self.cfg.MODEL.NAME in ["tcp","maple"]:
            print("Turning off gradients in both the image and the text encoder")
            name_to_update = "prompt_learner"
            for name, param in self.model.named_parameters():
                if name_to_update not in name:
                    param.requires_grad_(False)
                else:
                    print(name)

            self.model.to(self.device)
            # NOTE: only give prompt_learner to the optimizer
            self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
            self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
            
        elif self.cfg.MODEL.NAME == "fedpgp":
            print("Turning off gradients in both the image and the text encoder")
            for name, param in self.model.named_parameters():
                if "prompt_learner" not in name:
                    param.requires_grad_(False) 

            if cfg.DATASET.NAME == "ImageNet":
                self.device = torch.device("cuda:0")
                # device0 = torch.device("cuda:0")
                device1 = torch.device("cuda")
                self.model.to(self.device)
                self.model.text_encoder.to(device1)
                self.model.text_encoder = nn.DataParallel(self.model.text_encoder)
            else:
                self.model.to(self.device)
 
            self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
            self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched) 
        
        elif self.cfg.MODEL.NAME == "fedram":
            print("Turning off gradients in both the image and the text encoder")
            # for name, param in :
            #     if "prompt_learner" not in name:
            #         param.requires_grad_(False)
            self.freeze(self.cfg.TRAINER.RAM.TRANSFER_TYPE) 
            optimizer = make_optimizer(self.cfg,self.model)
            scheduler  = make_scheduler(cfg, optimizer, len(self.train_loader))

            self.model.to(self.device)
            # NOTE: only give prompt_learner to the optimizer 
            # params = ([p for p in self.model.prompt_learner.parameters()])
            self.optim = optimizer
            self.sched = scheduler
            self.register_model("prompt_learner", self.model, self.optim, self.sched)
        
        else:
            print("Turning off gradients in both the image and the text encoder")
            for name, param in self.model.named_parameters():
                if "prompt_learner" not in name:
                    param.requires_grad_(False)
            enabled = set()
            num_trainable_params = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    enabled.add(name)
                    num_trainable_params += param.data.nelement()
            print(f"Number of trainable parameters: {num_trainable_params / 1e6:.2f}")
            print(f"Parameters to be updated for client {self.client_id}: {enabled}")
            self.model.to(self.device)
            # NOTE: only give prompt_learner to the optimizer 
            # params = ([p for p in self.model.prompt_learner.parameters()])
            self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
            self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
          
        num_trainable_params = 0    
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
                num_trainable_params += param.data.nelement()
        print(f"Number of trainable parameters: {num_trainable_params / 1e6:.2f}")
        print(f"Parameters to be updated for client {self.client_id}: {enabled}")
            
    # for ram only.     
    def freeze(self, transfer_type):
        if self.model_teacher is not None:
            for name, param in self.model_teacher.named_parameters():
                param.requires_grad = False

        if transfer_type == "no_freeze":
            pass
        
        elif transfer_type == "freeze_all":
            for name, param in self.model.named_parameters():
                param.requires_grad = False
        
        elif transfer_type == "freeze_text":
            for name, param in self.model.named_parameters():
                if 'visual.' in name:
                    continue
                else:
                    param.requires_grad = False

        elif transfer_type == "Adapter":
            for name, param in self.model.named_parameters():
                if "adapter" in name or "embeds" in name:      #  embeds
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        elif "partial" in transfer_type:
            total_layer = len(self.model.visual.transformer.resblocks)
            partial_layer = int(transfer_type.split("-")[-1])
            if partial_layer > total_layer:
                raise NotImplementedError
            for name, param in self.model.named_parameters():
                find = False
                for l in range(total_layer-partial_layer, total_layer):
                    if "visual.transformer.resblocks.{}".format(l) in name:
                        param.requires_grad = True
                        find = True
                        break
                if not find:
                    param.requires_grad = False
                
        else:
            raise NotImplementedError


    def train(self, epoch):
        self.set_model_mode("train")
        losses = AverageMeter()
        mAP_batches = AverageMeter()
        batch_time = AverageMeter()
        dataname = self.data_name
        classnames = self.available_classes
        end = time.time()
        for idx, batch in enumerate(self.train_loader):
            images, labels = self.parse_batch(batch)
        
            loss, output = self.forward_backward([images, labels],dataname,classnames)
            self.model_backward_and_update(loss) 
 
            losses.update(loss.item(), images.size(0))
            Softmax = torch.nn.Softmax(dim=1)
            Sig = torch.nn.Sigmoid() 
            if output.dim() == 3:
                pred = Softmax(output.detach())[:, 1, :]
            else:
                pred = Sig(output.detach())
            mAP_value = mAP(labels.cpu().numpy(), pred.cpu().numpy())
            mAP_batches.update(mAP_value, images.size(0))
            # print(f"you take this time: {time.time()-end}")
            batch_time.update(time.time()-end)
            end = time.time()
            # print('Round: [{}/{}] Client: {} Train: [{0}/{1}]\t'
            #         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #         'Loss {losses.val:.2f} ({losses.avg:.2f})\t'
            #         'mAP {mAP_batches.val:.2f} ({mAP_batches.avg:.2f})'.format(num_round, self.cfg.OPTIM.MAX_EPOCH, self.client_id,
            #         idx, len(self.train_loader), batch_time=batch_time,
            #         losses=losses, mAP_batches=mAP_batches), flush=True)
        mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"********************************************** Client {self.client_id} train results For Epoch {epoch+1} **********************************************")
        print('Round: [{0}/{1}] Client: {2} Train: [{3}/{4}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {losses.val:.2f} ({losses.avg:.2f})\t'
                'mAP {mAP_batches.val:.2f} ({mAP_batches.avg:.2f}) Mem:{mems}'.format(epoch+1, self.cfg.OPTIM.MAX_EPOCH, self.client_id,
                idx+1, len(self.train_loader), batch_time=batch_time,
                losses=losses, mAP_batches=mAP_batches, mems=mem), flush=True)
         
        
        mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        if (epoch+1) == self.cfg.OPTIM.MAX_EPOCH:
            savestr = 'CLient-self test: Round: [{0}/{1}] Client: {2} \t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t Loss {losses.val:.2f} ({losses.avg:.2f})\t mAP {mAP_batches.val:.2f} ({mAP_batches.avg:.2f}) Mem {mems}'.format(epoch+1, self.cfg.OPTIM.MAX_EPOCH, self.client_id, batch_time=batch_time, losses=losses, mAP_batches=mAP_batches, mems=mem)
            f = open(self.cfg.TRAINER.SAVE_FILE.replace("AAAA",self.cfg.DATASET.NAME_SPACE[0]).replace("BBBB",self.cfg.MODEL.NAME),"a")
            f.write(savestr+'\n')
            f.close()
        
        print(f"save?{self.cfg.TRAINER.SAVE}")
        self.update_lr(epoch=epoch)
        if self.cfg.MODEL.NAME !="fedram":
            local_updates = self.model.prompt_learner.state_dict()
            # 不能save。太多了。
            if self.cfg.TRAINER.SAVE:
                save_dict = {'state_dict': self.model.prompt_learner.state_dict(),
                            'optimizer': self.optim.state_dict(),
                            'scheduler': self.sched.state_dict()
                            }
                save_dir = os.path.join(self.cfg.TRAINER.SAVE_DIR,"dataset_{}_method_{}_client_{}_partanno_{}/prompt_learner/".format(self.cfg.DATASET.NAME,self.cfg.MODEL.NAME,self.client_id,self.cfg.TRAINER.PA))
                save_checkpoint(save_dict, save_dir, True)
            return local_updates # 
        else:
            local_updates = self.model.state_dict()
            # 不能save。太多了。
            if self.cfg.TRAINER.SAVE:
                save_dict = {'state_dict': self.model.state_dict(),
                            'optimizer': self.optim.state_dict(),
                            'scheduler': self.sched.state_dict()
                            }
                save_dir = os.path.join(self.cfg.TRAINER.SAVE_DIR,"dataset_{}_method_{}_client_{}_partanno_{}/prompt_learner/".format(self.cfg.DATASET.NAME,self.cfg.MODEL.NAME,self.client_id,self.cfg.TRAINER.PA))
                save_checkpoint(save_dict, save_dir, True)
            return local_updates
 
 
    def load_meta(self, global_net):
        self.model.prompt_learner.load_state_dict(global_net)


    def load_ram(self, global_net):
        model_dict = self.model.state_dict()
        selected_dict = {name: param for name, param in global_net.state_dict().items()
                        if "adapter" in name or "embeds" in name}
        model_dict.update(selected_dict)
        self.model.load_state_dict(model_dict)
        # for name, param in global_net.named_parameters():
        #     if "adapter" in name or "embeds" in name:      #  embeds
        #         self.model.load_state_dict(param)
        #     else:
        #         pass

    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError 

    def forward_backward(self, batch, dataname,classnames):
        images, labels = batch
        if self.cfg.MODEL.NAME in ["maple", "dualcoop", "poscoop", "scpnet"]:
            output = self.model(images)
            # pdb.set_trace()
            loss = self.cfg.TRAINER.ML.LOSS_W * self.criterion(output, labels) 
        elif self.cfg.MODEL.NAME in ["tcp"]:
            output,loss1 = self.model(images, req=True)
            loss = self.cfg.TRAINER.ML.LOSS_W * self.criterion(output, labels) 
            loss += loss1
        elif self.cfg.MODEL.NAME in ["fedpgp"]:
            output, output_2 = self.model(images, req=True) 
            target = torch.zeros(output_2.size(0)).to(self.device).long()
            loss = self.cfg.TRAINER.ML.LOSS_W * (self.criterion(output, labels) + \
                self.cfg.TRAINER.FEDPGP.mu * F.cross_entropy(output_2, target))  
        elif self.cfg.MODEL.NAME in ["fedawa", "fedmvp", "fedmpt"]:
            output, kgscore = self.model(images,classnames, dataname, req=True)
            loss = self.cfg.TRAINER.ML.LOSS_W * (self.criterion(output, labels) + kgscore)
        elif self.cfg.MODEL.NAME in ["fedram"]:
            output,loss1 = self.model(images, labels, req=True)
            loss = loss1 
        else:
            output = self.model(images,classnames, dataname)
            loss = self.cfg.TRAINER.ML.LOSS_W * self.criterion(output, labels) 
        return loss, output

    def parse_batch(self, batch):
        input = batch["img"]
        label = batch["label"]
        # cname = batch["cname"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label # , cname

    def get_current_lr(self, names=None):
        # current_lr = self.sched.get_last_lr()
        # return current_lr[0]
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]
    def model_inference(self, input, classnames, dataname):
        # return self.model(input,classnames, dataname)
        return self.model(input, classnames, dataname)[0]
    
    
    @torch.no_grad()
    def validate(self, data_loader, model, cfg): 
        data_loader, dataname, classnames = data_loader
        Softmax = torch.nn.Softmax(dim=1)
        Sig = torch.nn.Sigmoid()
        # switch to evaluate mode
        model.eval() 
        
        PRED_CHOICES = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        HIGHEST = [0,0,0,0,0,0,0]
        for prec_c in PRED_CHOICES: 
            batch_time = AverageMeter()
            prec = AverageMeter()
            rec = AverageMeter()
            tp, fp, fn, tn, count = 0, 0, 0, 0, 0
            preds = []
            targets = []
            print(f"testing the threshold of {prec_c}") 
            with torch.no_grad():
                end = time.time()
                for i, batch in enumerate(data_loader):
                    images, target = self.parse_batch(batch)
                    target = target.max(dim=1)[0]
                    device = torch.device("cuda")
                    images = images.to(device)

                    # compute output
                    with autocast():
                        if self.cfg.MODEL.NAME in ["fedtpg","fedawa", "fedmvp", "fedmpt"]:
                            output = model(images,classnames,dataname)
                        else:
                            output = model(images) 
                    if output.dim() == 3:
                        output = Softmax(output).cpu()[:, 1]
                    else:
                        output = Sig(output).cpu()
                    # for mAP calculation
                    preds.append(output.cpu())
                    targets.append(target.cpu())
                    target = target.cpu()
                    # measure accuracy and record loss
                    pred = output.data.gt(prec_c).long() 
                    target = target.unsqueeze(dim=-1)
                    tp += (pred + target).eq(2).sum(dim=0)
                    fp += (pred - target).eq(1).sum(dim=0)
                    fn += (pred - target).eq(-1).sum(dim=0)
                    tn += (pred + target).eq(0).sum(dim=0)
                    count += images.size(0)
                
                    this_tp = (pred + target).eq(2).sum()
                    this_fp = (pred - target).eq(1).sum()
                    this_fn = (pred - target).eq(-1).sum()
                    this_tn = (pred + target).eq(0).sum()

                    this_prec = this_tp.float() / (
                            this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
                    this_rec = this_tp.float() / (
                            this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

                    prec.update(float(this_prec), images.size(0))
                    rec.update(float(this_rec), images.size(0))

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    # --------------V1---------------
                    # p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[
                    #                                                                     i] > 0 else 0.0
                    #     for i in range(len(tp))]
                    # r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[
                    #                                                                     i] > 0 else 0.0
                    #     for i in range(len(tp))]
                    # f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for
                    #     i in range(len(tp))]

                    # mean_p_c = sum(p_c) / len(p_c)
                    # mean_r_c = sum(r_c) / len(r_c)
                    # mean_f_c = sum(f_c) / len(f_c) 
                    
                    # ---------------V2----------------
                    p_c = []
                    r_c = []
                    f_c = []
                    for i in range(len(tp)): 
                        if (tp[i] + fp[i]) == 0 or (tp[i] + fn[i]) == 0:
                            continue
                        pi = float(tp[i]) / float(tp[i] + fp[i])
                        ri = float(tp[i]) / float(tp[i] + fn[i]) 
                        if (pi + ri) > 0:
                            fi = 2 * pi * ri / (pi + ri)
                        else:
                            fi = 0.0
                        p_c.append(pi * 100.0)
                        r_c.append(ri * 100.0)
                        f_c.append(fi * 100.0)

                    if len(p_c) > 0:
                        mean_p_c = sum(p_c) / len(p_c)
                        mean_r_c = sum(r_c) / len(r_c)
                        mean_f_c = sum(f_c) / len(f_c)
                    else: 
                        mean_p_c = 0.0
                        mean_r_c = 0.0
                        mean_f_c = 0.0 
                    

                    p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
                    r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
                    f_o = 2 * p_o * r_o / (p_o + r_o)

                    # if i % cfg.TRAIN.PRINT_FREQ == 0:
                    #     print('Test: [{0}/{1}]\t'
                    #         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    #         'Precision {prec.val:.2f} ({prec.avg:.2f})\t'
                    #         'Recall {rec.val:.2f} ({rec.avg:.2f}) \t '
                    #         'P_C {P_C:.2f} \t R_C {R_C:.2f} \t F_C {F_C:.2f} \t P_O {P_O:.2f} \t R_O {R_O:.2f} \t F_O {F_O:.2f}'.format(
                    #         i, len(data_loader), batch_time=batch_time,
                    #         prec=prec, rec=rec, P_C=mean_p_c, R_C=mean_r_c, F_C=mean_f_c, P_O=p_o, R_O=r_o, F_O=f_o), flush=True)

                mAP_score = mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())
                lili = [mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o, mAP_score]
                for idxxx in range(7):
                    if HIGHEST[idxxx] <= lili[idxxx]:
                        HIGHEST[idxxx] = lili[idxxx]

        torch.cuda.empty_cache()
        return HIGHEST # mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o, mAP_score
