import os, sys
sys.path.append(os.path.join("/".join(os.path.dirname(__file__).split("/")[:-1])))
import torch
from torch.utils.data import Dataset
from augment.aug import LargeAugmentation
from dlake_core.core import DlakeInterface

class Mnist(Dataset):
    def __init__(self, 
                 augment:LargeAugmentation = None,
                 subset = "test",
                 ratio = [0.8, 0.2],
                 categorical = True) -> None:
        super().__init__()
        self.augment: LargeAugmentation = augment
        self.categorical = categorical
        
        if subset == "train":
            self.ds, _ = DlakeInterface().get_dataset(dataset_name="mnist", subset="train").random_split(ratio)
        elif subset == "val":
            _, self.ds = DlakeInterface().get_dataset(dataset_name="mnist", subset="train").random_split(ratio)
        elif subset == "test":
            self.ds = DlakeInterface().get_dataset(dataset_name="mnist", subset="test")
        else:
            raise Exception(f"subset must be train, val or test but {subset} found instead")    
        
        self.num_class = 10
    
    def get_meta(self):
        return self.metadata
        
    def __getitem__(self, index):
        img = self.ds[index]["images"].data()["value"]
        label = self.ds[index]["labels"].data()["value"]
        
        if self.augment:
            img = self.augment(image = img)
        if self.categorical:
            one_hot_label = torch.zeros(10)
            one_hot_label[label.astype(int)] = 1
            label = one_hot_label
        
        return (torch.from_numpy(img), label)
    
    def __len__(self):
        return self.ds.num_samples

if __name__ == "__main__":
    mnist = Mnist(augment=LargeAugmentation(
        ran_crop = 0,
        ran_crop_width = 10,
        ran_crop_height = 10
    ))
    
    img, label = mnist[0]
    print(img.shape)
    print(label)