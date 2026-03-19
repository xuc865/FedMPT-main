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
from yacs.config import CfgNode as CN
def reset_cfg(cfg, args):
    cfg.EXP_NAME = args.exp_name
    cfg.DATASET.NAME_SPACE = [args.dataset] 
    cfg.DATASET.TESTNAME_SPACE = [args.dataset]
    cfg.TRAIN.SPLIT = 'all'
    cfg.TEST.SPLIT = 'all'
    
    cfg.DATASET.NAME = args.dataset
    if args.dataset == "voc":
        cfg.DATASET.NC = 20
    elif args.dataset == "coco":
        cfg.DATASET.NC = 80
    elif args.dataset == "nus":
        cfg.DATASET.NC = 81
    elif args.dataset == "object":
        cfg.DATASET.NC = 365
    elif args.dataset == "multiscene":
        cfg.DATASET.NC = 36
    elif args.dataset == "mlrsnet":
        cfg.DATASET.NC = 60
    else:
        raise NotImplementedError

    cfg.DATASET.ROOT = args.root
    cfg.DATASET.NUM_SHOTS = args.num_shots
    cfg.OUTPUT_DIR = args.output_dir
    cfg.RESUME = args.resume
    cfg.SEED = args.seed
    cfg.MODEL.BACKBONE.NAME = args.backbone
    cfg.OPTIM.MAX_EPOCH = args.num_epoch
    cfg.OPTIM.LR = args.lr


    cfg.MODEL.D_CTX = args.depth_ctx
    cfg.MODEL.N_CTX = args.n_ctx
    cfg.MODEL.DEPTH = args.model_depth
    cfg.MODEL.NAME = args.model_name

    cfg.DATALOADER.TRAIN.BATCH_SIZE = args.batch_size
    cfg.TRAIN.NUM_CLASS_PER_CLIENT = args.num_cls_per_client
    cfg.TRAIN.AVAIL_PERCENT = args.avail_percent 
    
    cfg.TRAINER = CN()
    cfg.TRAINER.ML = CN()
    cfg.TRAINER.ML.LOSS_W = 0.02
    cfg.TRAINER.ML.STUN = args.stun
    cfg.TRAINER.ML.LAT = args.lat
    cfg.TRAINER.ML.TEMP = args.temp
    cfg.TRAINER.ML.ALLOW_RESUME = args.allow_resume
    cfg.TRAINER.ML.COND = args.cond
    cfg.TRAINER.ML.CLS = args.cls
    cfg.TRAINER.ML.ZSL = args.zsl
    cfg.TRAINER.ML.THRE = 0.5
    cfg.TRAINER.ML.NUM_CLUSTERS = args.num_clusters
    cfg.TRAINER.ML.N_CTX_POS = 16
    cfg.TRAINER.ML.N_CTX_NEG = 16
    cfg.TRAINER.ML.CSC = False
    cfg.TRAINER.ML.POSITIVE_PROMPT_INIT = ""
    cfg.TRAINER.ML.NEGATIVE_PROMPT_INIT = ""
    cfg.TRAINER.ML.ASL_GAMMA_NEG = args.neg
    cfg.TRAINER.ML.ASL_GAMMA_POS = args.pos
    if args.pa != 0:
        cfg.TRAINER.SAVE_FILE = "PATHPDBB/PA/PDBB_AAAA_BBBB.txt"
    elif args.avail_percent != 1:
        cfg.TRAINER.SAVE_FILE = "PATHPDBB/CA/PDBB_AAAA_BBBB.txt"
    elif args.neda == True:
        cfg.TRAINER.SAVE_FILE = "PATHPDBB/ASLA/PDBB_AAAA_BBBB.txt"
    elif args.zsl == "gzsl":
        cfg.TRAINER.SAVE_FILE = "PATHPDBB/GZSL/PDBB_AAAA_BBBB.txt"
    elif args.zsl is not None:
        cfg.TRAINER.SAVE_FILE = "PATHPDBB/ZSLGZSL/PDBB_AAAA_BBBB.txt"
    else:
        cfg.TRAINER.SAVE_FILE = "PATHPDBB/PDBB_AAAA_BBBB.txt" 
    cfg.TRAINER.SAVE_DIR = "PATH/remote/PDBB/outputs/" 
    cfg.TRAINER.PA = args.pa
    cfg.TRAINER.SAVE = args.saving
    
    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    
    cfg.TRAINER.FEDPGP = CN()
    cfg.TRAINER.FEDPGP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.FEDPGP.CSC = False  # class-specific context
    cfg.TRAINER.FEDPGP.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.FEDPGP.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.FEDPGP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.FEDPGP.BOTTLENECK = 4
    cfg.TRAINER.FEDPGP.N = 2 # number of prompts
    cfg.TRAINER.FEDPGP.FEATURE = False
    cfg.TRAINER.FEDPGP.mu = 1
    cfg.TRAINER.FEDPGP.temp = 0.5

    cfg.TRAINER.COOP_MLC = CN()
    cfg.TRAINER.COOP_MLC.N_CTX_POS = 16
    cfg.TRAINER.COOP_MLC.N_CTX_NEG = 16
    cfg.TRAINER.COOP_MLC.CSC = False
    cfg.TRAINER.COOP_MLC.POSITIVE_PROMPT_INIT = ""
    cfg.TRAINER.COOP_MLC.NEGATIVE_PROMPT_INIT = ""
    cfg.TRAINER.COOP_MLC.ASL_GAMMA_NEG = 2
    cfg.TRAINER.COOP_MLC.ASL_GAMMA_POS = 1
    
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.ALPHA = 1.0
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = False  # initialization words
    cfg.TRAINER.COOP.W = 1.0
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = False  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp
    
     
     
     
    cfg.TRAINER.RAM = CN() 
    # Transformer setting
    cfg.TRAINER.RAM.DROP_PATH = 0.1
    cfg.TRAINER.RAM.DROP_OUT = 0.0
    cfg.TRAINER.RAM.ATT_DROP_RATE = 0.0
    cfg.TRAINER.RAM.TRANSFORMER_TYPE = 'None' 

    # JPM Parameter
    cfg.TRAINER.RAM.JPM = False
    cfg.TRAINER.RAM.SHIFT_NUM = 5
    cfg.TRAINER.RAM.SHUFFLE_GROUP = 2
    cfg.TRAINER.RAM.DEVIDE_LENGTH = 4
    cfg.TRAINER.RAM.RE_ARRANGE = True

    # SIE Parameter
    cfg.TRAINER.RAM.SIE_COE = 3.0
    cfg.TRAINER.RAM.SIE_CAMERA = False
    cfg.TRAINER.RAM.SIE_VIEW = False
 
    # Use BN
    cfg.TRAINER.RAM.BN = False
    # Number of head
    cfg.TRAINER.RAM.NUM_HEAD = 8
    # Loss type
    cfg.TRAINER.RAM.LOSS_TYPE = 'BCE'  # 'ASL'
    # ema model
    cfg.TRAINER.RAM.USE_EMA = False
    cfg.TRAINER.RAM.EMA_DECAY = 0.9997
    # load pretrain 
    cfg.TRAINER.RAM.LOAD = False
    # text encoder
    cfg.TRAINER.RAM.TEXT_ENCODER = 'CLIP' 
    cfg.TRAINER.RAM.TEXT_CTX = 4     # middle layers, 一般不用
    cfg.TRAINER.RAM.PROMPT_CSC = False

    # transfer type
    cfg.TRAINER.RAM.TRANSFER_TYPE = "Adapter"

    # SAA
    cfg.TRAINER.RAM.SAA_LAYER = [-1]

    # loc region pooling
    cfg.TRAINER.RAM.LOC_STRIDE_SIZE = 4
    cfg.TRAINER.RAM.LOC_KERNEL_SIZE = 4

    # Adapter 
    cfg.TRAINER.RAM.VISION_ADAPT = 8
    cfg.TRAINER.RAM.KERNEL_SIZE = 3

    # temperature
    cfg.TRAINER.RAM.TEMPERATURE = 0.002

    # new prototype
    cfg.TRAINER.RAM.NUM_NEW_PROTOTYPE = 10

    # OT reg
    cfg.TRAINER.RAM.OT_REG = 0.1
    cfg.TRAINER.RAM.OT_REGSC = 0.05

    # Adapter ratio
    cfg.TRAINER.RAM.ADAPTER_RATIO = 0.2
    cfg.TRAINER.RAM.SOLVER = CN()
    cfg.TRAINER.RAM.SOLVER.OPTIMIZER_NAME = 'AdamW'
    cfg.TRAINER.RAM.SOLVER.BASE_LR = 5e-5
    cfg.TRAINER.RAM.SOLVER.IMS_PER_BATCH = 32
    cfg.TRAINER.RAM.SOLVER.LARGE_FC_LR = False
    cfg.TRAINER.RAM.SOLVER.LOG_PERIOD = 1000
    cfg.TRAINER.RAM.SOLVER.WEIGHT_DECAY = 1e-4
    cfg.TRAINER.RAM.SOLVER.MAX_EPOCHS = 6
    cfg.TRAINER.RAM.SOLVER.EVAL_PERIOD = 3  
    cfg.TRAINER.RAM.SOLVER.SAVE_MODEL = False 
    
    # Number of max epoches FOR SCHEDULER
    cfg.TRAINER.RAM.SOLVER.SCHEDULER_MAX_EPOCHS = 60
    # Number of max epoches FOR SCHEDULER
    cfg.TRAINER.RAM.SOLVER.SCHEDULER_MAX_ITER = 1000000
    # Base learning rate 
    # Factor of learning bias
    cfg.TRAINER.RAM.SOLVER.BIAS_LR_FACTOR = 1
    # Factor of learning bias
    cfg.TRAINER.RAM.SOLVER.SEED = 42
    # Momentum
    cfg.TRAINER.RAM.SOLVER.MOMENTUM = 0.9
    # Margin of triplet loss
    cfg.TRAINER.RAM.SOLVER.MARGIN = 0.3
    # Learning rate of SGD to learn the centers of center loss
    cfg.TRAINER.RAM.SOLVER.CENTER_LR = 0.5
    # Balanced weight of center loss
    cfg.TRAINER.RAM.SOLVER.CENTER_LOSS_WEIGHT = 0.0005  
    cfg.TRAINER.RAM.SOLVER.WEIGHT_DECAY_BIAS = 0.0001
    cfg.TRAINER.RAM.SOLVER.WEIGHT_DECAY_SGD = 0.0001

    # decay rate of learning rate
    cfg.TRAINER.RAM.SOLVER.GAMMA = 0.1
    # decay step of learning rate
    cfg.TRAINER.RAM.SOLVER.STEPS = (40, 70)
    # warm up factor
    cfg.TRAINER.RAM.SOLVER.WARMUP_FACTOR = 0.01
    #  warm up epochs
    cfg.TRAINER.RAM.SOLVER.WARMUP_EPOCHS = 3
    # method of warm up, option: 'constant','linear'
    cfg.TRAINER.RAM.SOLVER.WARMUP_METHOD = "linear"

    cfg.TRAINER.RAM.SOLVER.COSINE_MARGIN = 0.5
    cfg.TRAINER.RAM.SOLVER.COSINE_SCALE = 30

    # epoch number of saving checkpoints
    cfg.TRAINER.RAM.SOLVER.CHECKPOINT_PERIOD = 10  
    # Classification Threshold
    # Loss type for contrastive
    cfg.TRAINER.RAM.SOLVER.THRESH = 0.5

    # Label smoothing
    cfg.TRAINER.RAM.SOLVER.LABEL_SMOOTHING = False

    # LR sheduler iter (TGPT imple)
    cfg.TRAINER.RAM.SOLVER.GAMMA = 0.1
    cfg.TRAINER.RAM.SOLVER.LR_SCHEDULER = "cosine"
    cfg.TRAINER.RAM.SOLVER.STEPSIZE = 1000

    # aslloss param
    cfg.TRAINER.RAM.SOLVER.GAMMA_NEG = 2
    cfg.TRAINER.RAM.SOLVER.GAMMA_POS = 0
    cfg.TRAINER.RAM.SOLVER.CLIP = 0.

    # twloss param
    cfg.TRAINER.RAM.SOLVER.TP = 4.
    cfg.TRAINER.RAM.SOLVER.TN = 1.

    # save the middle output, for visualization
    cfg.TRAINER.RAM.SOLVER.VERBOSE = False

    # iter training
    cfg.TRAINER.RAM.SOLVER.MAX_ITER = 12800
    cfg.TRAINER.RAM.SOLVER.WARMUP_ITER = 200
    cfg.TRAINER.RAM.SOLVER.BASE_LR_SGD = 0.001

    # KD loss weight
    cfg.TRAINER.RAM.SOLVER.KDLOSS_WEIGHT = 1.

    # Text batch
    cfg.TRAINER.RAM.SOLVER.TEXT_BATCH_SIZE = 80

    # debug mode
    cfg.TRAINER.RAM.SOLVER.DEBUG = False

    # zero-shot testing
    cfg.TRAINER.RAM.SOLVER.ZS_TEST = False

    # sample text
    cfg.TRAINER.RAM.SOLVER.SAMPLE_TEXT = False
    
    
    
    cfg.TRAINER.RAM.PRETRAIN_CHOICE = 'imagenet' 
    cfg.TRAINER.RAM.NECK = 'bnneck'  
    cfg.TRAINER.RAM.ID_LOSS_TYPE = 'softmax'
    cfg.TRAINER.RAM.ID_LOSS_WEIGHT = 1.0
    cfg.TRAINER.RAM.TRIPLET_LOSS_WEIGHT = 1.0

    cfg.TRAINER.RAM.METRIC_LOSS_TYPE = 'triplet' 
    cfg.TRAINER.RAM.DIST_TRAIN = False   
    cfg.TRAINER.RAM.COS_LAYER = False 
    cfg.TRAINER.RAM.IF_LABELSMOOTH = 'off'
    cfg.TRAINER.RAM.IF_WITH_CENTER = 'no'
    cfg.TRAINER.RAM.NO_MARGIN = True
    cfg.TRAINER.RAM.STRIDE_SIZE = [16, 16]
    cfg.TRAINER.RAM.BACKBONE = 'ViT-B/16'
    cfg.TRAINER.RAM.LOSS_TYPE = 'MMC'
    cfg.TRAINER.RAM.DEPTH_VISION = [9,10,11]
    cfg.TRAINER.RAM.DEPTH_TEXT = [6,7,8,9,10,11]
    cfg.TRAINER.RAM.PROMPT_CSC = False
    cfg.TRAINER.RAM.TRANSFER_TYPE = "Adapter"
    cfg.TRAINER.RAM.SAA_LAYER = [12, -1]
    cfg.TRAINER.RAM.USE_EMA = True
    cfg.TRAINER.RAM.KERNEL_SIZE = 3
    cfg.TRAINER.RAM.TEMPERATURE = 2e-4
    cfg.TRAINER.RAM.OT_REGSC = 0.05