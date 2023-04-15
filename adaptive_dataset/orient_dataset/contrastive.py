import os, sys
sys.path.append(os.path.join("/".join(os.path.dirname(__file__).split("/")[:-1])))
import torch
from torch.utils.data import Dataset
from augment.aug import LargeAugmentation
from core_dataset import CoreDataset

class CLDataset(Dataset, CoreDataset):
    def __init__(self, 
                 dataset_name:str,
                 subset:str,
                 n_views:int = 2,
                 augment:LargeAugmentation = None,
                 ratio: list[int, int] = [0.8, 0.2],
                 access_method = "stream"
                 ) -> None:
        super().__init__(dataset_name=dataset_name, 
                         subset=subset, 
                         access_method=access_method, 
                         ratio=ratio)
        
        self.n_views = n_views
        self.augment: LargeAugmentation = augment
        
        if self.n_views != 2:
            raise ValueError("n_views must be 2. The other case have not been support yet")
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        img = self.ds[index]["images"].data()["value"]
        # label = self.ds[index]["labels"].data()["value"]
        
        imgs = [self.augment(img) for i in range(self.n_views)]
        
        return torch.from_numpy(imgs)