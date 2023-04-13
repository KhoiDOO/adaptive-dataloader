import os, sys
import albumentations as A
from albumentations import DualTransform
import numpy as np
import cv2
from typing import *

class Transfer(DualTransform):
    def __init__(self, always_apply: bool = True, p: float = 1):
        super().__init__(always_apply, p)
    
    def apply(self, img, h_start=0, w_start=0, **params):
        return img
    
    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

class LargeAugmentation:
    def __init__(self, *args, **kwargs) -> None:
        
        self.kwargs = kwargs
    
        self.transform = A.Compose([
            A.GaussianBlur(
                # p=self.kwargs["gau_blur_p"]
                ) if "gau_blur" in self.kwargs.keys() else Transfer(),
            A.RandomCrop(
                width=self.kwargs["ran_crop_width"], 
                height=self.kwargs["ran_crop_height"]
                ) if "ran_crop" in self.kwargs.keys() else Transfer(),
            A.RGBShift(
                
                ) if "rgbshift" in self.kwargs.keys() else Transfer(),
            A.RandomBrightnessContrast(
                
                ) if "ran_bright_cons" in self.kwargs.keys() else Transfer(),
            A.Flip(
                
                ) if "flip" in self.kwargs.keys() else Transfer(),
            A.HorizontalFlip(
                
                ) if "hor_flip" in self.kwargs.keys() else Transfer(),
            A.VerticalFlip(
                
                ) if "ver_flip" in self.kwargs.keys() else Transfer(),
            A.SafeRotate(
                
                ) if "safe_rot" in self.kwargs.keys() else Transfer(),
            A.Normalize(
                
                ) if "normalize" in self.kwargs.keys() else Transfer()
        ])

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        tranformed =  self.transform(
            image = kwds["image"] if "image" in kwds.keys() else None,
            # mask = kwds["mask"] if "mask" in kwds.keys() else None,
            # masks = kwds["masks"] if "masks" in kwds.keys() else None
        )
        
        return tranformed["image"]
        
        if "mask" in tranformed.keys():
            return (tranformed["image"], tranformed["mask"])
        elif "mask" in tranformed.keys():
            return (tranformed["image"], tranformed["masks"])
        else:
            return tranformed["image"]

class SegAumentation(LargeAugmentation):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        tranformed =  self.transform(
            image = kwds["image"] if "image" in kwds.keys() else None,
            mask = kwds["mask"] if "mask" in kwds.keys() else None,
            masks = kwds["masks"] if "masks" in kwds.keys() else None
        )
        
        if "mask" in tranformed.keys():
            return (tranformed["image"], tranformed["mask"])
        elif "mask" in tranformed.keys():
            return (tranformed["image"], tranformed["masks"])
        else:
            return tranformed["image"]

if __name__ == "__main__":
    aug = SegAumentation(
        ran_crop = 0,
        ran_crop_width = 128,
        ran_crop_height = 128,
        gau_blur = 0,
        gau_blur_p = 1
    )
    
    img_path = "/media/asus/working/git/adaptive-dataloader/test/CHGastro_Abnormal_013.png"
    mask_path = "/media/asus/working/git/adaptive-dataloader/test/CHGastro_Abnormal_013_ROI.png"
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mask = cv2.imread(mask_path)
    
    output = aug(image = img, mask = mask)
    
    cv2.imshow(winname="img", mat=img)
    cv2.imshow(winname="mask", mat=mask)
    cv2.imshow(winname="aug_img", mat=output[0])
    cv2.imshow(winname="aug_mask", mat=output[1])
    cv2.waitKey(0)