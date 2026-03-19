import os
from os.path import join
import torch
import json
from tqdm import tqdm
import pdb
from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image
from dataloader.utils import *
from mldata.abstract_cluster import ClusterMLDataset
from mldata.cls_to_names import nuswide_classes
import random
import numpy as np
class NUSWIDE(ClusterMLDataset, DatasetBase):
    NAME = "NUSWIDE"
    def __init__(self, set_id, cfg, available_classes, traintest, clip_model):
        ClusterMLDataset.__init__(self, cfg)

        self.dataset_dir = os.path.join(cfg.DATASET.ROOT, "NUSWIDE", "raw")
        self.class_names = nuswide_classes 
        self.sss = traintest
        
        if cfg.TRAINER.ML.ZSL is None:
            self.image_dir = os.path.join(self.dataset_dir, "Flickr")
            self.cls_name_list = self.read_name_list(join(self.dataset_dir, 'Concepts81.txt'), False)
            assert traintest is not None
            if traintest == "train": 
                self.im_name_list = self.read_name_list(join(self.dataset_dir, 'ImageList/TrainImagelist.txt'), False)
            else:
                self.im_name_list = self.read_name_list(join(self.dataset_dir, 'ImageList/TestImagelist.txt'), False) # 107859
                
            path_labels = os.path.join(self.dataset_dir, 'TrainTestLabels')
            num_classes = len(self.class_names)
            test_labels = defaultdict(list)
            for i in tqdm(range(num_classes)):
                YD = "Train" if traintest == "train" else "Test"
                file_ = os.path.join(path_labels, 'Labels_'+self.class_names[i]+f'_{YD}.txt') # 107859
                cls_labels = []
                with open(file_, 'r') as f:
                    for j, line in enumerate(f):
                        tmp = line.strip()
                        if tmp == '1':
                            test_labels[j].append(i) # 第几个类属于哪些样本
            
            self.data_list = []
            for i, name in tqdm(enumerate(self.im_name_list)):
                img_path=self.image_dir + '/' + '/'.join(name.split('\\'))
                label=test_labels[i]
                label = list(set(label))  
                # if i >=4000:
                #     pdb.set_trace()
                #     print(img_path, [self.class_names[jj] for jj in label])  
                if label:
                    if len(label) >= 5: 
                        if cfg.TRAINER.PA != 0:
                            label = [x for x in label if random.random() > cfg.TRAINER.PA]
                        self.data_list.append([img_path, label])
        else:
            if traintest == 'train':
                self.train_file = os.path.join("PATHPDBB/FedTPG-main/labs", 'train_81_filtered.json')
                train, class2idx, name_train = self._load_dataset(self.dataset_dir, self.train_file, shuffle=True)
                self.data_list = train 
            elif cfg.TRAINER.ML.ZSL == "zsl" and traintest == "test":
                self.test_file = os.path.join("PATHPDBB/FedTPG-main/labs", 'test_81_filtered.json')
                test, _, name_test = self._load_dataset(self.dataset_dir, self.test_file, shuffle=False)
                self.data_list = test 
            elif cfg.TRAINER.ML.ZSL == "gzsl" and traintest == "test":
                self.test_file_gzsl = os.path.join("PATHPDBB/FedTPG-main/labs", 'test_1006_filtered.json')
                test_gzsl, _, _ = self._load_dataset(self.dataset_dir, self.test_file_gzsl, shuffle=False)
                self.data_list = test_gzsl
            else:
                raise NotImplementedError
        
        self.id_cls_mapping = self.cluster(clip_model) # 个超类，每个超类含有一些数据映射  
        darts = self.subsample_classes(self.id_cls_mapping, available_classes)
        DatasetBase.__init__(self, train=darts, val=darts, test=darts, nc=len(self.class_names)) 
 
    def _load_dataset(self, data_dir, annot_path, shuffle=True):
        out_data = []
        with open(annot_path) as f:
            annotation = json.load(f)
            classes = self.class_names # sorted(annotation['classes']) if names is None else names
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            images_info = annotation['images']
            img_wo_objects = 0
            for img_info in images_info:
                labels_idx = list()
                rel_image_path, img_labels = img_info 
                full_image_path = rel_image_path.replace("/mnt/nas/TrueNas1/tanhao/NUS_WIDE/Flickr","PATH/NUSWIDE/raw/Flickr/") 
      
                labels_idx = [class_to_idx[lbl] for lbl in img_labels if lbl in class_to_idx]
                labels_idx = list(set(labels_idx))
                # transform to one-hot 
                # if not (len(labels_idx) >= 3): 
                #     continue
                onehot = np.zeros(len(classes), dtype=int)
                onehot[labels_idx] = 1
                assert full_image_path
                if not labels_idx:
                    img_wo_objects += 1
                out_data.append((full_image_path, onehot))
        if img_wo_objects:
            print(f'WARNING: there are {img_wo_objects} images without labels and will be treated as negatives')
        if shuffle:
            random.shuffle(out_data)
        return out_data, class_to_idx, classes


    def read_name_list(self, path, if_split=True):
        ret = []
        with open(path, 'r') as f:
            for line in f:
                if if_split:
                    tmp = line.strip().split(' ')
                    ret.append(tmp[0])
                else:
                    tmp = line.strip()
                    ret.append(tmp)
        return ret
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]

        image = Image.open(open(img_path, "rb")).convert("RGB")
        target = torch.LongTensor(label)

        return image, img_path, target