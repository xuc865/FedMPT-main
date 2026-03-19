import csv
import os
import os.path
from dataloader.utils import *
from mldata.abstract_cluster import ClusterMLDataset
import numpy as np
import torch
import torch.utils.data as data
from mldata.cls_to_names import mlrsnet_classes
from PIL import Image
import pdb


CATES = [
    "airplane",
    "airport",
    "bareland",
    "baseball_diamond",
    "basketball_court",
    "beach",
    "bridge",
    "chaparral",
    "cloud",
    "commercial_area",
    "dense_residential_area",
    "desert",
    "eroded_farmland",
    "farmland",
    "forest",
    "freeway",
    "golf_course",
    "ground_track_field",
    "harbor&port",
    "industrial_area",
    "intersection",
    "island",
    "lake",
    "meadow",
    "mobile_home_park",
    "mountain",
    "overpass",
    "park",
    "parking_lot",
    "parkway",
    "railway",
    "railway_station",
    "river",
    "roundabout",
    "shipping_yard",
    "snowberg",
    "sparse_residential_area",
    "stadium",
    "storage_tank",
    "swimmimg_pool",
    "tennis_court",
    "terrace",
    "transmission_tower",
    "vegetable_greenhouse",
    "wetland",
    "wind_turbine"
]

class MLRSNet(ClusterMLDataset, DatasetBase):
    NAME = "MLRSNet"
    def __init__(self, set_id, cfg, available_classes, traintest, clip_model): 
        ClusterMLDataset.__init__(self, cfg)
        self.path_dataset = os.path.join(cfg.DATASET.ROOT, 'MLRSNet') 

        # define filename of csv file
        catesets = None
        if traintest == "train":
            catesets = CATES[:len(CATES)//2]
        elif traintest == "test":
            catesets = CATES[len(CATES)//2:]
        
        data_list = []
        for cate in catesets:
            file_csv = os.path.join(self.path_dataset, "Labels", cate+'.csv')  
            header = True
            self.class_names = mlrsnet_classes 
            num_categories = len(self.class_names)
            #print('[dataset] read', filename)
            with open(file_csv, 'r') as f:
                reader = csv.reader(f)
                rownum = 0
                for row in reader:
                    if header and rownum == 0:
                        header = row
                    else:
                        name = row[0] # os.path.join(self.dataset_dir, "images", )
                        gt = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                        if gt.sum() >= 7:
                            gt = torch.from_numpy(gt)
                            if cfg.TRAINER.PA != 0:
                                mask = (gt == 1) & (torch.rand_like(gt, dtype=torch.float) > cfg.TRAINER.PA)
                                gt = gt * mask.to(gt.dtype)
                            item = (name, gt) 
                            data_list.append(item)
                    rownum += 1
                
        self.data_list = data_list
        self.id_cls_mapping = self.cluster(clip_model)  
        darts = self.subsample_classes(self.id_cls_mapping, available_classes) 
        DatasetBase.__init__(self, train=darts, val=darts, test=darts, nc=len(self.class_names))

    def __getitem__(self, index):
        path, target = self.data_list[index]
        clsn = path.split("_0")[0]
        path = os.path.join(self.path_dataset, 'Images', clsn, path)
        img = Image.open(open(path, "rb")).convert('RGB') 
        # target = torch.LongTensor(target)

        return img, path, target

    def __len__(self):
        return len(self.data_list)

    def get_number_classes(self):
        return len(self.class_names)
