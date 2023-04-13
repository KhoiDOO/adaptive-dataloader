import os, sys
sys.path.append(os.path.join("/".join(os.path.dirname(__file__).split("/")[:-1])))
import torch
from torch.utils.data import Dataset
from augment.aug import LargeAugmentation
from glob import glob
import cv2 as cv
from PIL import Image
from utils import *
import numpy as np

class InfoExtract(Dataset):
    def __init__(self, 
                 main_data_dir:str = None, 
                 label2index: dict = None,
                 subset = "train") -> None:
        super().__init__()
        self.main_data_dir = main_data_dir
        self.label2index = label2index
        self.__check_main_data_dir()
        self.__check_label2index()
        
        self.subset = subset
        self.file_ext = [".jpg", ".png", ".jpeg"]
        
        self.data_path = self.main_data_dir + f"/*/{self.subset}"
        
        files_grabbed = []
        for ext in self.file_ext:
            files_grabbed.extend(glob(self.data_path + f"/*{ext}"))
        self.img_paths = files_grabbed
        self.json_paths = [x[:-4] + ".json" for x in self.img_paths]
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        json_path = self.json_paths[index]
        
        boxes, texts, labels, additions = parse_json(json_path)
        
        bb_clusters = sort_bbs(bbs=np.array(boxes).reshape(-1, 8), texts=texts, labels=labels)
        
        boxes = []
        texts = []
        labels = []
        
        for cluster in bb_clusters:
            boxes += [x[[0,1,3,4]].tolist() for x in cluster[0]]
            texts += cluster[2]
            labels += cluster[3]
        
        label_indices = [self.label2index[label] for label in labels]
        
        encode = self.process(img, texts, boxes=boxes, word_labels=label_indices, truncation=True, stride = 128, 
                    padding="max_length", max_length=512, 
                    # return_overflowing_tokens=True, 
                    # return_offsets_mapping=True, 
                    # return_tensors="pt"
                    )
        
        return encode

    def __check_main_data_dir(self):
        if self.main_data_dir is None:
            raise Exception("mian_data_dir cannot be None")
        elif not isinstance(self.main_data_dir, str):
            raise Exception(f"main data dir must be a string but found {type(self.main_data_dir)}")
        elif not os.path.exists(self.main_data_dir):
            raise Exception(f"the folder {self.main_data_dir} does not exist")
    
    def __check_label2index(self):
        if self.label2index is None:
            raise Exception("label2index cannot be None")
        elif not isinstance(self.label2index, dict):
            raise Exception(f"label2index data dir must be a string but found {type(self.main_data_dir)}")
        elif not os.path.exists(self.label2index):
            raise Exception(f"the folder {self.label2index} does not exist")