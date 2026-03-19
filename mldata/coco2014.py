import os
from os.path import join
import torch
import json
from tqdm import tqdm
from dataloader.utils import *
from mldata.abstract_cluster import ClusterMLDataset
from torch.utils.data import Dataset
from PIL import Image
from mldata.cls_to_names import coco2014_classes
from pycocotools.coco import COCO
import random
import numpy as np
import pdb
class COCO2014(ClusterMLDataset, DatasetBase):
    NAME = "COCO2014"
    def __init__(self, set_id, cfg, available_classes, traintest, clip_model): # set_id, dataset_dir, transform 
        ClusterMLDataset.__init__(self, cfg)

        self.dataset_dir = os.path.join(cfg.DATASET.ROOT, "coco")
        self.class_names = coco2014_classes 
        self.sss = traintest
        if cfg.TRAINER.ML.ZSL is None:
            if traintest == 'train':
                coco2014 = os.path.join(self.dataset_dir, "annotations/instances_train2014.json")
            elif traintest == 'test':
                coco2014 = os.path.join(self.dataset_dir, "annotations/instances_val2014.json")
            else:
                raise NotImplementedError()

            self.coco_data = COCO(coco2014)
            self.ids_val = self.coco_data.getImgIds() 
            categories = self.coco_data.loadCats(self.coco_data.getCatIds())
            categories.sort(key=lambda x: x['id']) 
            classes = {}
            coco_labels = {}
            coco_labels_inverse = {}
            for c in categories:
                coco_labels[len(classes)] = c['id']
                coco_labels_inverse[c['id']] = len(classes)
                classes[c['name']] = len(classes) 
            self.data_list = []
            for idx, imgid in tqdm(enumerate(self.ids_val)):
                ip = "train2014" if traintest == 'train' else "val2014"
                img_path = self.dataset_dir + f"/{ip}/{self.coco_data.loadImgs(imgid)[0]['file_name']}"
                label = self.load_annotations(coco_labels_inverse, self.coco_data, None, imgid, filter_tiny=False)
                label = list(set(label))
                if label:
                    if len(label) >= 5:
                        if cfg.TRAINER.PA != 0:
                            label = [x for x in label if random.random() > cfg.TRAINER.PA]
                        self.data_list.append([img_path, label]) 
        else: 
            if traintest == 'train':
                self.train_file = os.path.join("PATHPDBB/FedTPG-main/labs", 'train_48_filtered.json')
                train, class2idx, name_train = self._load_dataset(self.dataset_dir, self.train_file, shuffle=True)
                self.data_list = train 
            elif cfg.TRAINER.ML.ZSL == "zsl" and traintest == "test":
                self.test_file = os.path.join("PATHPDBB/FedTPG-main/labs", 'test_17_filtered.json')
                test, _, name_test = self._load_dataset(self.dataset_dir, self.test_file, shuffle=False)
                self.data_list = test 
            elif cfg.TRAINER.ML.ZSL == "gzsl" and traintest == "test":
                self.test_file_gzsl = os.path.join("PATHPDBB/FedTPG-main/labs", 'test_65_filtered.json')
                test_gzsl, _, _ = self._load_dataset(self.dataset_dir, self.test_file_gzsl, shuffle=False)
                self.data_list = test_gzsl
            else:
                raise NotImplementedError
        
        self.id_cls_mapping = self.cluster(clip_model)  
        darts = self.subsample_classes(self.id_cls_mapping, available_classes) 
        DatasetBase.__init__(self, train=darts, val=darts, test=darts, nc=len(self.class_names))              

    def _load_dataset(self, data_dir, annot_path, shuffle=True, names=None):
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
                full_image_path = rel_image_path.replace("/mnt/nas/TrueNas1/tanhao/COCO","PATH/coco") 
                if self.sss == "train":
                    full_image_path = full_image_path.replace("trainval2014", "train2014")
                elif self.sss == "test":
                    full_image_path = full_image_path.replace("trainval2014", "val2014") 
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


    def load_annotations(self, coco_labels_inverse, coco_, img_idlist, image_index, filter_tiny=True):
        tmp_id = image_index if (img_idlist is None) else img_idlist[image_index]
        annotations_ids = coco_.getAnnIds(imgIds=tmp_id, iscrowd=False)
        annotations = []

        if len(annotations_ids) == 0:
            return annotations

        coco_annotations = coco_.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            if filter_tiny and (a['bbox'][2] < 1 or a['bbox'][3] < 1):
                continue
            annotations += [coco_labels_inverse[a['category_id']]]

        return annotations 

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]

        image = Image.open(open(img_path, "rb")).convert("RGB") 
        target = torch.LongTensor(label)

        return image, img_path, target