import os
from augment.aug import LargeAugmentation
from dlake_core.core import DlakeInterface

os.environ["DEEPLAKE_DOWNLOAD_PATH"] = "~/data/"

support_dataset = {
    "mnist" : 0,
    "imagenet" : 1,
    "coco" : 1,
    "celeb-a" : 1,
    "cifar10" : 0, 
    "cifar100" : 0,
    "fashion-mnist" : 0,
    "kmnist" : 0
}

class CoreDataset:
    def __init__(self, 
                 dataset_name:str = None, 
                 subset:str = None,
                 access_method:str = None,
                 ratio: list[int, int] = None) -> None:
        self.dataset_name = dataset_name
        self.subset = subset
        self.access_method = access_method
        self.ratio = ratio
        
        if self.dataset_name not in list(support_dataset.keys()):
            raise Exception("The chose dataset is currently not supported")

        if support_dataset[self.dataset_name] == 0:
            if self.subset == "train":
                self.ds, _ = DlakeInterface().get_dataset(
                    dataset_name=self.dataset_name, 
                    subset="train",
                    access_method=self.access_method).random_split(self.ratio)
            elif self.subset == "val":
                _, self.ds = DlakeInterface().get_dataset(
                    dataset_name=self.dataset_name, 
                    subset="train",
                    access_method = self.access_method).random_split(self.ratio)
            elif self.subset == "test":
                self.ds = DlakeInterface().get_dataset(
                    dataset_name=self.dataset_name, 
                    subset="test",
                    access_method = self.access_method)
            else:
                raise Exception(f"subset must be train, val or test but {subset} found instead")    
        else:
            self.ds = DlakeInterface().get_dataset(
                dataset_name=self.dataset_name, 
                subset=self.subset,
                access_method = self.access_method)