from data.base_dataset import BaseDataset, Rescale_fixed, Normalize_image
from data.image_folder import make_dataset, make_dataset_test

import os
from tqdm import tqdm
from PIL import Image

import torch
import torchvision.transforms as transforms


class AlignedDataset(BaseDataset):
    def __init__(self, opt, dataset_type="train"):
        self.opt = opt

        if dataset_type == "train":
            self.image_dir = opt.image_folder
            self.mask_dir = opt.mask_folder
        else:
            self.image_dir = opt.image_folder_test
            self.mask_dir = opt.mask_folder_test

        # self.df_path = opt.df_path
        self.width = opt.fine_width
        self.height = opt.fine_height

        # for rgb imgs

        self.image_files = os.listdir(self.image_dir)
        self.mask_files = os.listdir(self.mask_dir)

        transforms_list = []
        transforms_list += [transforms.ToTensor()]
        transforms_list += [Normalize_image(opt.mean, opt.std)]
        self.transform_rgb = transforms.Compose(transforms_list)

        self.dataset_size = len(self.image_files)

    def __getitem__(self, index):
        # load images ad masks
        idx = index

        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = img_path.replace("images", "annotations").replace(".jpg", ".png")

        # img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.width, self.height), resample=Image.BICUBIC)
        image_tensor = self.transform_rgb(img)

        mask = Image.open(mask_path).convert("L")
        mask = transforms.ToTensor()(mask) * 255

        mask_replaced = torch.zeros_like(mask)

        mask_replaced[mask == 1] = 1
        mask_replaced[mask == 4] = 2
        mask_replaced[mask == 6] = 3
        mask_replaced[mask == 9] = 4

        mask_replaced[(mask != 1) & (mask != 4) & (mask != 6) & (mask != 9)] = 0

        mask_replaced = transforms.Resize((self.width, self.height))(mask_replaced)
        
        target_tensor = torch.as_tensor(mask_replaced, dtype=torch.int64)

        return image_tensor, target_tensor

    def __len__(self):
        return self.dataset_size

    def name(self):
        return "AlignedDataset"