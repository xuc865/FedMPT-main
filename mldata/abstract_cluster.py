from torch.utils.data import Dataset
import torch
import clip
from sklearn.cluster import KMeans
from tqdm import tqdm  
import os
import pickle 
import numpy as np
import pdb
import torch.nn.functional as F
class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        classname (str): class name.
    """

    def __init__(self, impath="", label=0, classname=""):
        # assert isinstance(impath, str)

        self._impath = impath
        self._label = label
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def classname(self):
        return self._classname

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
    
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class ClusterMLDataset(Dataset):
    def __init__(self,cfg):
        self.cfg = cfg
        self.data_name = self.NAME
        self.mode = "label" 
        
    @torch.no_grad()
    def cluster(self,clip):
        path = ""
        if self.cfg.TRAINER.ML.ZSL is None:
            path = f"clusters/{self.mode}/{self.NAME}_{self.cfg.TRAINER.ML.NUM_CLUSTERS}.pkl"
        else:
            path = f"clusters/{self.mode}/{self.NAME}_{self.cfg.TRAINER.ML.NUM_CLUSTERS}_PE{self.cfg.TRAINER.ML.ZSL}.pkl"
            
        if not os.path.exists(path):
            data = self.cluster_with_clip(clip, self.class_names, self.cfg.TRAINER.ML.NUM_CLUSTERS)
            f = open(path, "wb")
            data = pickle.dump(data, f)
            f.close()
        else:
            f = open(path, "rb")
            data = pickle.load(f)
            f.close()
        return data
             
    @torch.no_grad()
    def cluster_with_clip(self, clip_model, class_names, num_clusters=30, device="cuda"):
        """
        Use CLIP zero-shot classification features on ML dataset to form clusters.
        """
        clip_model.eval().to(device) 
        # 1. Build class text embeddings
        text_tokens = clip.tokenize([f"a photo of a {c}" for c in class_names]).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # 2. Extract image features and zero-shot logits
        all_features = [] 
        trans = _transform(224)
        mode = self.mode 
        pathh = None
        if self.cfg.TRAINER.ML.ZSL is None:
            pathh = f"clusters/{mode}/{self.NAME}_features.pkl"
        else:
            pathh = f"clusters/{mode}/{self.NAME}_features_PE{self.cfg.TRAINER.ML.ZSL}.pkl"
            
        features_to_cluster = None
        if not os.path.exists(pathh):
            if mode == "label":
                for i, (image, _, target) in enumerate(tqdm(self, desc="Extracting CLIP features for clustering....")):  
                    if target.shape[-1] != len(self.class_names):
                        one_hot = F.one_hot(target, num_classes=len(self.class_names)).sum(dim=0,keepdim=True).clamp(0, 1)
                        all_features.append(one_hot.cpu()) # 20
                    else: 
                        all_features.append(target.unsqueeze(dim=0).cpu()) # 20
                    # all_indices.append(i)
                all_features = torch.cat(all_features, dim=0).float()
                probs = all_features.softmax(dim=-1)
                features_to_cluster = probs.cpu().numpy()  # [N, 20]
            else:
                for i, (image, _, target) in enumerate(tqdm(self, desc="Extracting CLIP features for clustering....")): 
                    image = trans(image).unsqueeze(0).to(device)
                    image_features = clip_model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    all_features.append(image_features.cpu())
                    # all_indices.append(i)
                all_features = torch.cat(all_features, dim=0)
                logits_per_image = (all_features @ text_features.T)
                probs = logits_per_image.softmax(dim=-1)
                features_to_cluster = probs.cpu().numpy()  # [N, 20]
            
            np.save(pathh, features_to_cluster) 
        else:
            features_to_cluster = np.load(pathh) 
          
        print("clustering...........")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        print("end clustering")
        cluster_ids = kmeans.fit_predict(features_to_cluster) 
        # 5. Build mapping structures
        # cluster_assignments = {i: [] for i in range(num_clusters)} 
        # for idx, cid in enumerate(cluster_ids):
        #     cluster_assignments[cid].append(idx)  
        return cluster_ids.tolist() # cluster_assignments, 
 
    def subsample_classes(self, hyps, available_classes,relabel=True):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """   
        dataset = self 
        dataset_new, cname_new = [], []
        print("subsampling classes...............") 
        for (img, img_path, cls), hyp_cls in tqdm(zip(dataset, hyps), desc="Subsamping...."): 
            if not hyp_cls in available_classes:
                continue
            item_new = Datum(
                impath=img_path,
                label=self.to_one_hot(cls, len(self.class_names)) # relabeler[hyp_cls]
            )
            dataset_new.append(item_new) 
        print(f"this dasnew {len(dataset_new)}")
        return dataset_new

    def to_one_hot(self, cls, C):  
        if cls.shape[-1] == C:
            return cls.to(torch.float32)
        one_hot = torch.zeros(C, dtype=torch.float32) 
        one_hot[cls] = 1.0
        return one_hot