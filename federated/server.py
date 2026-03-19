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
from fvcore.nn import FlopCountAnalysis
from functools import partial 
import pdb
import time
import numpy as np
import datetime
from tqdm import tqdm
from model import *
from dataloader.dm_federated import TestDataManager 
from federated.utils import *
import copy
from federated.client import Client
import math
import random 
import torch
import time
from utilss.helper import AverageMeter, mAP, calc_F1
from torch.cuda.amp import autocast
from federated.base_trainer import TrainerBase
from mldata.cls_to_names import *
from convclip import clip
from convclip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def convert_params_to_value(params):
    if params[0] == -1:
        return [-1]    # not using
    elif params[-1] == -1:
        return list(range(params[0]))    # continuous N layers
    else:
        return params

def load_clip_to_cpu_zs(cfg,zero_shot=False): 
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    if zero_shot:
        saa_layer = [12, -1] if "ViT-B" in backbone_name else [24, -1]
        saa_layer = convert_params_to_value(saa_layer)
        design_details = {
            "depth_vision": [-1],
            "depth_text": [-1],
            "SAA_layer": saa_layer
        }
        print("Build zero-shot CLIP Model")
    else:
        depth_vision = convert_params_to_value(cfg.MODEL.DEPTH_VISION)
        depth_text = convert_params_to_value(cfg.MODEL.DEPTH_TEXT)
        saa_layer = convert_params_to_value(cfg.MODEL.SAA_LAYER)
        design_details = {
            "depth_vision": depth_vision,
            "vision_adapt": cfg.MODEL.VISION_ADAPT,
            "depth_text": depth_text,
            "text_ctx": cfg.MODEL.TEXT_CTX, 
            "SAA_layer": saa_layer,
            "kernel_size": cfg.MODEL.KERNEL_SIZE
        }
        print("Build CLIP Model")
    
    model = clip.build_model_ram(state_dict or model.state_dict(), (224, 224), design_details)
    model.visual.SAA_replace()

    return model.float()




def load_clip_to_cpu(cfg,zero_shot=False):
    if cfg.MODEL.NAME == "maple":
        
        backbone_name = cfg.MODEL.BACKBONE.NAME
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")
        design_details = {"trainer": 'MaPLe',
                        "vision_depth": 0,
                        "language_depth": 0, "vision_ctx": 0,
                        "language_ctx": 0,
                        "maple_length": cfg.TRAINER.MAPLE.N_CTX}
        model = clip.build_model_maple(state_dict or model.state_dict(), design_details)

        return model
    elif cfg.MODEL.NAME == "fedram":
        backbone_name = cfg.MODEL.BACKBONE.NAME
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        if zero_shot:
            saa_layer = [12, -1] if "ViT-B" in backbone_name else [24, -1]
            saa_layer = convert_params_to_value(saa_layer)
            design_details = {
                "depth_vision": [-1],
                "depth_text": [-1],
                "SAA_layer": saa_layer
            }
            print("Build zero-shot CLIP Model")
        else:
            depth_vision = convert_params_to_value(cfg.TRAINER.RAM.DEPTH_VISION)
            depth_text = convert_params_to_value(cfg.TRAINER.RAM.DEPTH_TEXT)
            saa_layer = convert_params_to_value(cfg.TRAINER.RAM.SAA_LAYER)
            design_details = {
                "depth_vision": depth_vision,
                "vision_adapt": cfg.TRAINER.RAM.VISION_ADAPT,
                "depth_text": depth_text,
                "text_ctx": cfg.TRAINER.RAM.TEXT_CTX, 
                "SAA_layer": saa_layer,
                "kernel_size": cfg.TRAINER.RAM.KERNEL_SIZE
            }
            print("Build CLIP Model")
        
        model = clip.build_model_ram(state_dict or model.state_dict(), (224, 224), design_details)
        model.visual.SAA_replace()

        return model.float()
  
    elif cfg.MODEL.NAME == "tcp":
        backbone_name = cfg.MODEL.BACKBONE.NAME
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url) 
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None 
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu") 
        model = clip.build_model_tcp(state_dict or model.state_dict()) 
        return model
    elif cfg.MODEL.NAME in ["fedmvp", "fedmpt"]:
        backbone_name = cfg.MODEL.BACKBONE.NAME
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu") 
        model = clip.build_model_fedmvp(state_dict or model.state_dict()) 
        return model

    else:
        backbone_name = cfg.MODEL.BACKBONE.NAME
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")
        model = clip.build_model_conv_proj(state_dict or model.state_dict(), cfg)

        return model

class Server(TrainerBase):
    # expand the trainer to the FL scenarios

    def __init__(self, cfg, nobuild=False):
        super().__init__()
        self.device = torch.device("cuda")

        seed = cfg.SEED
        if seed >= 0:
            np.random.seed(seed)
            random.seed(seed)
            # Set the random seed for PyTorch
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = os.path.join(cfg.OUTPUT_DIR,cfg.EXP_NAME,cfg.MODEL.NAME,str(cfg.TRAINER.ML.NUM_CLUSTERS)+"_"+str(cfg.TRAINER.PA),str(cfg.SEED))
        self.cfg = cfg
        self.build_model()

        cfg.defrost()

        self.evaluator = Classification(cfg)
        self.clients = []
        self.init_server(cfg,nobuild=nobuild)

        self.cfg = cfg


    def build_model(self):
        cfg = self.cfg

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        self.model_name = cfg.MODEL.NAME
        self.zsmodel = load_clip_to_cpu_zs(cfg,zero_shot=True)
 
        classsnames = ""
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
        self.classnames = classsnames
        print("Building custom CLIP")
        nam = cfg.MODEL.NAME
        if nam == 'fedtpg':
            self.model = FedTPG(cfg, clip_model, classnames=classsnames, device = self.device)
        elif nam == "dualcoop":
            self.model = DualCoop(cfg, classnames=classsnames, clip_model=clip_model)
        elif nam == "poscoop":
            self.model = PositiveCoop(cfg, classnames=classsnames, clip_model=clip_model)
        elif nam == "scpnet":
            self.model = SCPNet(cfg, classnames=classsnames, clip_model=clip_model)
        elif nam == 'coop':
            self.model = CoOpCLIP(cfg, clip_model,classnames=classsnames,device = self.device)
        elif nam == 'fedawa':
            self.model = FedAWACLIP(cfg, clip_model,classnames=classsnames, device = self.device)
        elif nam == 'fedmvp':
            self.model = FedMVPCLIP(cfg, clip_model,classnames=classsnames, device = self.device)
        elif nam == 'fedmpt':
            self.model = FedMPTCLIP(cfg, clip_model,classnames=classsnames,device = self.device)
        elif nam == 'maple':
            self.model = MapleCLIP(cfg, classsnames, clip_model)
        elif nam == 'tcp':
            self.model = TCPCLIP(cfg, classsnames, clip_model)
        elif nam == 'fedpgp':
            self.model = FedPGPCLIP(cfg, classsnames, clip_model)
        elif nam == 'vlp':
            self.model = VLPCLIP(cfg, clip_model,device = self.device)
        elif nam == 'fedram':
            self.model = FedRAMCLIP(cfg, clip_model,classsnames,[], self.zsmodel, device = self.device)

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
        print(f"Parameters to be updated for server: {enabled}")
        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        if nam !=  'fedram':    
            self.register_model("prompt_learner", self.model.prompt_learner)
        else:
            self.register_model("prompt_learner", self.model)

        self.clip_model = clip_model 

    def init_server(self, cfg,nobuild=False):

        num_class_per_client = cfg.TRAIN.NUM_CLASS_PER_CLIENT
        available_datasets = cfg.DATASET.NAME_SPACE
        cluster_num = cfg.TRAINER.ML.NUM_CLUSTERS
 
        dataset_classnum = cluster_num
        client_id,last_num_clients = 0,0

        for dataname in available_datasets:

            m = dataset_classnum
            all_cls_idx = np.arange(m)
            num_client_dataset = np.around(m/num_class_per_client).astype(int)
            if num_client_dataset==0:
                num_client_dataset = 1
            current_num_clients = last_num_clients+num_client_dataset
            while client_id< current_num_clients:

                if client_id==current_num_clients-1:
                    available_cls = all_cls_idx[(client_id - last_num_clients) * num_class_per_client:]
                else:
                    available_cls = all_cls_idx[(client_id-last_num_clients)*num_class_per_client:(client_id-last_num_clients+1)*num_class_per_client]

                client = Client(cfg, len(self.clients),dataname,available_cls,self.clip_model,nobuild=nobuild,zsmodel=self.zsmodel)

                self.clients.append(client)
                client_id+=1
            last_num_clients = current_num_clients

        self.num_clients = len(self.clients)

        print(f'total number of clients:{self.num_clients}')
        
    def distribute(self, idx): 
        if self.cfg.MODEL.NAME!="fedram":
            self.clients[idx].load_meta(self.meta_net_glob.state_dict())
        else:
            self.clients[idx].load_ram(self.meta_net_glob)

    def train(self):
        self.before_train()
        if self.cfg.MODEL.NAME!="fedram":
            self.meta_net_glob = copy.deepcopy(self.model.prompt_learner)
        else:
            self.meta_net_glob = copy.deepcopy(self.model)
            
        if self.cfg.MODEL.NAME in ["maple", "dualcoop", "poscoop", "fedtpg", "fedpgp", "tcp", "fedmvp", "fedmpt", "scpnet"]:
            for epoch in range(self.start_epoch, self.max_epoch):
                self.epoch = epoch
                num_selected = max(int(self.cfg.TRAIN.AVAIL_PERCENT * self.num_clients), 1)
                idxs_users = np.random.choice(range(len(self.clients)), num_selected, replace=False)
                w_glob = None
                for idx in idxs_users:
                    self.distribute(idx)
                    w_local = self.clients[idx].train(epoch)
                    if w_glob is None:
                        w_glob = copy.deepcopy(w_local)
                    else:
                        for k in w_glob.keys():
                            w_glob[k] +=w_local[k]
                
                for k in w_glob.keys():
                    w_glob[k] = torch.div(w_glob[k], num_selected)
                self.meta_net_glob.load_state_dict(w_glob, strict=False)
                
                if self.cfg.TRAINER.ML.STUN != 0 and epoch % self.cfg.TRAINER.ML.STUN == 0:
                    self.model.prompt_learner.load_state_dict(self.meta_net_glob.state_dict()) 
                    self.test(self.cfg.TEST.SPLIT, self.clip_model)
            self.model.prompt_learner.load_state_dict(self.meta_net_glob.state_dict()) 
        
        elif self.cfg.MODEL.NAME in ["fedram"]:
            for epoch in range(self.start_epoch, self.max_epoch):
                self.epoch = epoch
                num_selected = max(int(self.cfg.TRAIN.AVAIL_PERCENT * self.num_clients), 1)
                idxs_users = np.random.choice(range(len(self.clients)), num_selected, replace=False)
                w_glob = None
                for idx in idxs_users:
                    self.distribute(idx)
                    w_local = self.clients[idx].train(epoch)
                    if w_glob is None:
                        w_glob = copy.deepcopy(w_local)
                    else:
                        for k in w_glob.keys():
                            w_glob[k] +=w_local[k]
                
                for k in w_glob.keys():
                    w_glob[k] = torch.div(w_glob[k], num_selected)
                self.meta_net_glob.load_state_dict(w_glob, strict=False)
                
                if self.cfg.TRAINER.ML.STUN != 0 and epoch % self.cfg.TRAINER.ML.STUN == 0:
                    self.model.prompt_learner.load_state_dict(self.meta_net_glob.state_dict()) 
                    self.test(self.cfg.TEST.SPLIT, self.clip_model) 
            self.model.load_state_dict(self.meta_net_glob.state_dict())
            
            
        elif self.cfg.MODEL.NAME in ["fedawa"]:
            for epoch in range(self.start_epoch, self.max_epoch):
                self.epoch = epoch
                num_selected = max(int(self.cfg.TRAIN.AVAIL_PERCENT * self.num_clients), 1)
                idxs_users = np.random.choice(range(len(self.clients)), num_selected, replace=False)
                w_glob = None
                alist = []
                for idx in idxs_users:
                    self.distribute(idx)
                    w_local = self.clients[idx].train(epoch)
                    alist.append(w_local)
                    if w_glob is None:
                        w_glob = copy.deepcopy(w_local)
                    else:
                        for k in w_glob.keys():
                            w_glob[k] +=w_local[k]
                
                if len(idxs_users) != 1:
                    w_realg = copy.deepcopy(w_glob)
                    for k in w_glob.keys():
                        w_glob[k] = torch.div(w_glob[k], num_selected)
                    # filter those deviated:
                    THRE = 0.3 
                    SCORE = []
                    for item in alist:
                        for k in item.keys():
                            # pdb.set_trace()
                            SCORE.append(F.cosine_similarity(item[k], w_glob[k]).mean())
                    indexes = top_percent_indices(SCORE, THRE)
                    for index in indexes:
                        for k in w_glob.keys():
                            w_realg[k] -= alist[index][k]
                    for k in w_realg.keys():
                        w_realg[k] = torch.div(w_realg[k], num_selected - len(indexes))
                    w_glob = w_realg        
                        
                self.meta_net_glob.load_state_dict(w_glob, strict=False) 
                if self.cfg.TRAINER.ML.STUN != 0 and epoch % self.cfg.TRAINER.ML.STUN == 0:
                    self.model.prompt_learner.load_state_dict(self.meta_net_glob.state_dict()) 
                    self.test(self.cfg.TEST.SPLIT, self.clip_model) 
            self.model.prompt_learner.load_state_dict(self.meta_net_glob.state_dict())
        else:
            for epoch in range(self.start_epoch, self.max_epoch):
                self.epoch = epoch
                num_selected = max(int(self.cfg.TRAIN.AVAIL_PERCENT * self.num_clients), 1)
                idxs_users = np.random.choice(range(len(self.clients)), num_selected, replace=False)
                for idx in idxs_users:
                    self.distribute(idx)
                    _ = self.clients[idx].train(epoch)
                if self.cfg.TRAINER.ML.STUN != 0 and epoch % self.cfg.TRAINER.ML.STUN == 0:
                    self.model.prompt_learner.load_state_dict(self.meta_net_glob.state_dict()) 
                    self.test(self.cfg.TEST.SPLIT, self.clip_model)
                
        self.after_train()

    def model_inference(self, input,classnames, dataname):
        # return self.model(input,classnames, dataname)
        return self.model(input,classnames, dataname)[0]

    def parse_batch(self, batch):
        input = batch["img"]
        label = batch["label"]
        # cname = batch["cname"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label # , cname 
 
    def before_train(self):
        directory = self.output_dir
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        if self.cfg.TRAINER.ML.ALLOW_RESUME:
            self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        writer_dir = os.path.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_train(self):
        print("Finish training")
        last_epoch = (self.epoch + 1) == self.max_epoch
        if last_epoch:
            self.save_model(self.epoch, self.output_dir)
        do_test = not self.cfg.TEST.NO_TEST
        print("I reach epoch {}/{}".format(self.epoch + 1, self.max_epoch))
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            # eval_based on each dataset
            self.local_test()
            self.test(self.cfg.TEST.SPLIT, self.clip_model)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))

        print(f"Elapsed: {elapsed}")

    @torch.no_grad()
    def test(self,split, clip_model):
        """A generic testing pipeline."""
        self.set_model_mode("eval")

        dm = TestDataManager(self.cfg, self.cfg.DATASET.NAME, clip_model, available_classes=None)
        data_loader = dm.test_loader

        summing = [0,0,0,0,0,0,0]
        p_c, r_c, f_c, p_o, r_o, f_o, mAP_score = self.validate((data_loader, self.cfg.DATASET.NAME, self.classnames), self.model, self.cfg)
        for idxx, item in enumerate([p_c, r_c, f_c, p_o, r_o, f_o, mAP_score]):
            summing[idxx] += item
        sum_res = []
        for item in summing:
            sum_res.append(item)
        print(f"********************************************** Global test results **********************************************") 
        savebb = 'Server Global Test TEMP{} LAT{} COND:{} CLS:{} avail_percent{} PA{}: P_C {:.2f} \t R_C {:.2f} \t F_C {:.2f} \t P_O {:.2f} \t R_O {:.2f} \t F_O {:.2f} \t mAP {:.2f}'.format(self.cfg.TRAINER.ML.TEMP, self.cfg.TRAINER.ML.LAT, self.cfg.TRAINER.ML.COND, self.cfg.TRAINER.ML.CLS, self.cfg.TRAIN.AVAIL_PERCENT,self.cfg.TRAINER.PA,*sum_res)
        print(savebb)
        f = open(self.cfg.TRAINER.SAVE_FILE.replace("AAAA",self.cfg.DATASET.NAME_SPACE[0]).replace("BBBB",self.cfg.MODEL.NAME),"a")
        f.write(savebb+'\n')
        f.close()  
         
    @torch.no_grad()
    def local_test(self):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        acc_dict = {}

        for i, client in enumerate(self.clients):
            self.evaluator.reset()
            print(f"Evaluate on the *{str(i)}th* client of {client.data_name}")
            classnames = client.available_classes
            dataname = client.data_name
            test_loader = client.test_loader 
            
            p_c, r_c, f_c, p_o, r_o, f_o, mAP_score = self.validate((test_loader, dataname, classnames), client.model, self.cfg)
            print(f"********************************************** Local test results **********************************************")
            savebb = 'Cluster {}s PA{} Server Local Test: CLIENT {} Test: P_C {:.2f} \t R_C {:.2f} \t F_C {:.2f} \t P_O {:.2f} \t R_O {:.2f} \t F_O {:.2f} \t mAP {:.2f}'.format(self.cfg.TRAINER.ML.NUM_CLUSTERS, self.cfg.TRAINER.PA, i, p_c, r_c, f_c, p_o, r_o, f_o, mAP_score)
            print(savebb) 
            f = open(self.cfg.TRAINER.SAVE_FILE.replace("AAAA",self.cfg.DATASET.NAME_SPACE[0]).replace("BBBB",self.cfg.MODEL.NAME),"a")
            f.write(savebb+'\n')
            f.close()

            # results = self.evaluator.evaluate()
            # acc= list(results.values())[0]

            # if dataname not in acc_dict:
            #     acc_dict[dataname]= [acc]
            # else:
            #     acc_dict[dataname].append(acc)
        # acc_list = []
        # for key in acc_dict.keys():
        #     acc_list.append(np.mean(acc_dict[key]))
        #     print(f"avg acc of {key}: {np.mean(acc_dict[key])}")
        # print(f"avg local accuracy: {np.mean(acc_list)}")

    def reset_flops(self,module):
        if hasattr(module, '__flops__'):
            del module.__flops__
        if hasattr(module, '__params__'):
            del module.__params__
        for child in module.children():
            self.reset_flops(child)


    @torch.no_grad()
    def validate(self, data_loader, model, cfg): 
        data_loader, dataname, classnames = data_loader
        Softmax = torch.nn.Softmax(dim=1) 
        Sig = torch.nn.Sigmoid()
        # switch to evaluate mode
        model.eval() 
        self.reset_flops(model)
 
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
                    
                    if torch.cuda.is_available():
                        device = torch.device("cuda")
                    else:
                        device = torch.device("cpu")
                    images = images.to(device)
 
                    # compute output
                    with autocast():
                        if self.cfg.MODEL.NAME in ["fedtpg","fedawa", "fedmvp", "fedmpt"]:
                            output = model(images,classnames,dataname)
                        else:
                            output = model(images) 
                            
                    # x = torch.randn(1, 3, 224, 224).cuda()
                    # flops = FlopCountAnalysis(model, x)
                    # print(flops.total() / 1e9, "GFLOPs")
                            
                             
                    if output.dim() == 3:
                        output = Softmax(output).cpu()[:, 1]
                    else:
                        output = Sig(output).cpu()
                    target = target.cpu()
    
                    # for mAP calculation
                    preds.append(output)
                    targets.append(target)
    
                    # measure accuracy and record loss
                    pred = output.data.gt(prec_c).long() # cfg.TRAINER.ML.THRE 
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

                    p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[
                                                                                        i] > 0 else 0.0
                        for i in range(len(tp))]
                    r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[
                                                                                        i] > 0 else 0.0
                        for i in range(len(tp))]
                    f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for
                        i in range(len(tp))]

                    mean_p_c = sum(p_c) / len(p_c)
                    mean_r_c = sum(r_c) / len(r_c)
                    mean_f_c = sum(f_c) / len(f_c)

                    p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
                    r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
                    f_o = 2 * p_o * r_o / (p_o + r_o)

                    # if i % cfg.TRAINER.ML.PRINT_FREQ == 0:
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



    @torch.no_grad()
    def tupian(self, image_path, model):
        dataname, classnames = "coco", ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        Softmax = torch.nn.Softmax(dim=1)
        Sig = torch.nn.Sigmoid()
        # switch to evaluate mode
        model.eval()  
        n_px = 224
        transform = Compose([
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
          ])

        # --- load image ---
        img = Image.open(image_path).convert('RGB')
        x = transform(img)  # C,H,W
        x = x.unsqueeze(0)  # 1,C,H,W 
        with torch.no_grad(): 
            images = x 
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            images = images.to(device)

            # compute output
            with autocast():
                if False: # self.cfg.MODEL.NAME in ["fedtpg", "fedawa", "fedmvp", "fedmpt"]:
                    output = model(images, classnames, dataname)
                else:
                    output = model(images)
            if output.dim() == 3:
                output = Softmax(output).cpu()[:, 1]
            else:
                output = Sig(output).cpu() 
            aa = {}
            for a,b, in zip(classnames, output[0]): 
                aa[a] = round(b.item(), 4)
            print(sorted(aa.items(), key=lambda x: x[1], reverse=True))
        torch.cuda.empty_cache() 

def top_percent_indices(tensor_list, percent=0.30):
    values = [t.item() if isinstance(t, torch.Tensor) else float(t) for t in tensor_list]
    n = len(values)
    if n == 0:
        return []
    sorted_idx = sorted(range(n), key=lambda i: values[i])
    cut = int((1 - percent) * n)
    top_idxs = sorted_idx[cut:]
    return top_idxs

