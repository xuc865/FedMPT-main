import os
from os.path import join
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from mldata.abstract_cluster import ClusterMLDataset
from mldata.cls_to_names import voc2007_classes
from PIL import Image
from dataloader.utils import * 
import pdb
import random
class VOC2007(ClusterMLDataset, DatasetBase):
    NAME = "VOC2007"
    def __init__(self, set_id, cfg, available_classes, traintest, clip_model):
        ClusterMLDataset.__init__(self, cfg)
        
        dataset_dir = cfg.DATASET.ROOT
        assert traintest is not None
        if traintest == "train":
            self.dataset_dir = os.path.join(dataset_dir, "VOC2007/VOCtrainval2007/VOCdevkit/VOC2007")
            self.im_name_list = self.read_name_list(join(self.dataset_dir, 'ImageSets/Main/trainval.txt'))
        else:
            self.dataset_dir = os.path.join(dataset_dir, "VOC2007/VOCtest2007/VOCdevkit/VOC2007") 
            self.im_name_list = self.read_name_list(join(self.dataset_dir, f'ImageSets/Main/test.txt'))
            
        phase = traintest  
        self.image_dir = os.path.join(self.dataset_dir, "JPEGImages") 
        print('VOC2007 {} total {} images. '.format(traintest, len(self.im_name_list)))
        self.class_names = voc2007_classes
        self.id_cls_mapping = None
        
        test_data_imname2label = self.read_object_labels(self.dataset_dir, phase=phase)   
        self.data_list = []
        for i, name in enumerate(self.im_name_list):
            img_path = self.image_dir+'/{}.jpg'.format(name)
            label = test_data_imname2label[name]
            label = list(set(label)) 
            if label:
                if cfg.TRAINER.PA != 0 and traintest == "train":
                    label = [x for x in label if random.random() > cfg.TRAINER.PA]
                self.data_list.append([img_path, label]) 
            
        self.id_cls_mapping = self.cluster(clip_model)  
        darts = self.subsample_classes(self.id_cls_mapping, available_classes) 
        DatasetBase.__init__(self, train=darts, val=darts, test=darts, nc=len(self.class_names))
    
    def read_object_labels(self, path, phase):
        path_labels = os.path.join(path, 'ImageSets', 'Main')
        labeled_data = defaultdict(list)
        num_classes = len(self.class_names)

        for i in range(num_classes):
            file = os.path.join(path_labels, self.class_names[i] + '_' + phase + '.txt')
            data_ = self.read_image_label(file)

            for (name, label) in data_.items():
                if label == 1:
                    labeled_data[name].append(i)
        return labeled_data

    def read_image_label(self, file):
        data_ = dict()
        with open(file, 'r') as f:
            for line in f:
                tmp = line.strip().split(' ')
                name = tmp[0]
                label = int(tmp[-1])
                data_[name] = label
        return data_

    def read_name_list(self, path):
        ret = []
        with open(path, 'r') as f:
            for line in f:
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
