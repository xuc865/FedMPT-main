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

import torch
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
from torch.utils.data import Sampler
from dataloader.fed_datasets import *
import numpy as np
from clip.clip import _transform
import random
import torchvision.transforms as T
from tqdm import tqdm
from collections import defaultdict
import random
import math
from mldata import *
 
 
def build_data_loader(
        cfg,
        data_source,
        batch_size=64,
        tfm=None,
):
    dataset_wrapper = DatasetWrapper(data_source, transform=tfm)
    # Build sampler
    sampler = None 

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
    )
    assert len(data_loader) > 0

    return data_loader
 
 
# def build_dataset(set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False):
import torchvision.transforms as transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])

class TrainDataManager:

    def __init__(
            self,
            cfg,
            dataname,
            clip_model,
            available_classes=None
    ):
        assert available_classes is not None
        # transform = transforms.Compose([
        #     transforms.Resize(224, interpolation=BICUBIC),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     normalize])
        # Load dataset 
        if dataname == 'coco':
            dataset = COCO2014('coco2014', cfg, available_classes, traintest="train", clip_model=clip_model)
        elif dataname == 'voc':
            dataset = VOC2007('voc2007', cfg, available_classes, traintest="train", clip_model=clip_model)
        elif dataname == 'nus':
            dataset = NUSWIDE('nuswide', cfg, available_classes, traintest="train", clip_model=clip_model)
        elif dataname == 'multiscene':
            dataset = MultiScene('multiscene', cfg, available_classes, traintest="train", clip_model=clip_model)
        elif dataname == 'mlrsnet':
            dataset = MLRSNet('mlrsnet', cfg, available_classes, traintest="train", clip_model=clip_model)
        elif dataname == 'object':
            dataset = OBJECT365('object', cfg, available_classes, traintest="train", clip_model=clip_model)
            
        tfm = _transform(224)
        # Build train_loader

        train_loader = build_data_loader(
            cfg,
            data_source=dataset.train,
            batch_size=cfg.DATALOADER.TRAIN.BATCH_SIZE,
            tfm=tfm,
        )

        test_loader = build_data_loader(
            cfg,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm,
        )
        self.train_loader = train_loader
        self.available_classes = dataset.class_names
        self.data_name = dataset.data_name
        self.test_loader = test_loader




class TestDataManager:

    def __init__(
            self,
            cfg,
            dataname,
            clip_model,
            available_classes=None,
    ):
        dataset_classnum = cfg.TRAINER.ML.NUM_CLUSTERS
        all_cls_idx = np.arange(dataset_classnum) 
        if available_classes is None:
            available_classes = all_cls_idx # global test use
          
        if dataname == 'coco':
            dataset = COCO2014('coco2014', cfg, available_classes, traintest="test", clip_model=clip_model)
        elif dataname == 'voc':
            dataset = VOC2007('voc2007', cfg, available_classes, traintest="test", clip_model=clip_model)
        elif dataname == 'nus':
            dataset = NUSWIDE('nuswide', cfg, available_classes, traintest="test", clip_model=clip_model)
        elif dataname == 'multiscene':
            dataset = MultiScene('multiscene', cfg, available_classes, traintest="test", clip_model=clip_model)
        elif dataname == 'mlrsnet':
            dataset = MLRSNet('mlrsnet', cfg, available_classes, traintest="train", clip_model=clip_model)
        elif dataname == 'object':
            dataset = OBJECT365('object', cfg, available_classes, traintest="test", clip_model=clip_model)
            
        tfm = _transform(224)
        test_loader = build_data_loader(
            cfg,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm,
        ) 
             
        self.test_loader = test_loader 


class DatasetWrapper(TorchDataset):

    def __init__(self, data_source, transform=None):
        self.data_source = data_source
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        label = item.label
        try:
            img = Image.open(item.impath).convert("RGB")
        except:
            img = item.impath

        if self.transform is not None:
            img = self.transform(img)

        output = {
            "img": img,
            "label": label
        }
        return output

